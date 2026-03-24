# -*- coding: utf-8 -*-
"""
RAG System - FAISS (CPU) VERSION (Mistral required, performance tuned)
- FAISS CPU retrieval, embeddings on GPU if available
- Forces Mistral-7B-Instruct (cache-first -> download). If Mistral fails to load the script exits with clear guidance.
- Always uses the LLM (no retrieval-only fallback)
- Uses BitsAndBytes 4-bit quantization (if bitsandbytes installed)
- Avoids early_stopping param for causal models to remove warnings
Run with:
    python new_mistral_required.py
Prerequisites:
    pip install -U transformers sentence-transformers bitsandbytes sentencepiece faiss-cpu
    huggingface-cli login  # if model gated
    set HF_TOKEN=...       # or export on Linux/macOS
"""

import os
import time
import json
import traceback
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import textwrap
import pickle
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# -------------------------
# Configuration
# -------------------------
FAQ_PATH = "admission_faq.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE = "embeddings.npy"
PASSAGES_CACHE = "passages.pkl"
FAISS_INDEX_PATH = "faiss.index"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Reduce HF symlink noise on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -------------------------
# Startup logs
# -------------------------
print("=" * 72)
print("RAG SYSTEM - FAISS (CPU) VERSION (Mistral required, performance tuned)")
print("=" * 72)

gpu_available = torch.cuda.is_available()
print(f"GPU available for PyTorch: {gpu_available}")
if gpu_available:
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (torch): {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# -------------------------
# Utilities: load & split passages
# -------------------------
def load_and_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAQ file not found: {path}\nCreate it in the project dir.")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages = []
    for p in raw:
        if len(p) > 1200:
            parts = textwrap.wrap(p, width=800)
            passages.extend(parts)
        else:
            passages.append(p)
    return passages

print("\n[1/6] Loading FAQ passages...")
passages = load_and_split(FAQ_PATH)
print(f"✅ Loaded {len(passages)} passages.")

# -------------------------
# Embedding model (GPU if available)
# -------------------------
print("\n[2/6] Loading sentence-transformers embedding model...")
device = "cuda" if gpu_available else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL)
try:
    embed_model.to(device)
except Exception:
    pass
embed_model.eval()
print(f"✅ Embedding model ready on {device}")

# -------------------------
# Compute or load embeddings + FAISS CPU index
# -------------------------
if os.path.exists(EMBED_CACHE) and os.path.exists(PASSAGES_CACHE) and os.path.exists(FAISS_INDEX_PATH):
    print("\n[3/6] Loading cached embeddings and FAISS index (CPU)...")
    embeddings = np.load(EMBED_CACHE)
    with open(PASSAGES_CACHE, "rb") as f:
        passages = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("✅ Cache loaded.")
else:
    print("\n[3/6] Computing embeddings (this may take a moment)...")
    t0 = time.time()
    emb_tensor = embed_model.encode(
        passages,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=64,
        device=device
    )
    emb_tensor = emb_tensor / emb_tensor.norm(dim=1, keepdim=True).clamp(min=1e-9)
    embeddings = emb_tensor.cpu().numpy().astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    np.save(EMBED_CACHE, embeddings)
    with open(PASSAGES_CACHE, "wb") as f:
        pickle.dump(passages, f)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"⚡ Embeddings computed and FAISS index built in {time.time() - t0:.2f}s")

print("Using FAISS CPU index for search.")

# -------------------------
# Retrieval helpers
# -------------------------
def embed_query(q, model=embed_model):
    with torch.inference_mode():
        vec = model.encode([q], convert_to_tensor=True, device=device)
        vec = vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return vec.cpu().numpy().astype("float32")

def retrieve_top_k(query, k=3):
    qv = embed_query(query)
    D, I = index.search(qv, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append((int(idx), float(score), passages[idx]))
    return results

# -------------------------
# Mistral loader (cache-first -> download), required
# -------------------------
print(f"\n[4/6] Loading language model (cache-first): {MISTRAL_MODEL_ID}")
print("Loader will try cache first (no network). If absent, it may download and cache.")

def _attempt_load_mistral(local_only: bool, bnb_config: BitsAndBytesConfig):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_ID, use_fast=True, trust_remote_code=True,
                                                  token=hf_token, local_files_only=local_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_ID, use_fast=False, trust_remote_code=True,
                                                  token=hf_token, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL_ID,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=local_only
    )
    model.eval()
    return tokenizer, model

# BitsAndBytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)

m_tokenizer = None
m_model = None
mistral_loaded = False
mistral_source = "not attempted"

if gpu_available:
    # try cache-first
    try:
        m_tokenizer, m_model = _attempt_load_mistral(local_only=True, bnb_config=bnb_config)
        mistral_loaded = True
        mistral_source = "cache"
        print("✅ Mistral loaded from cache (no download).")
    except Exception as e_cache:
        print("• Mistral not found in cache or cache-load failed. Attempting download...")
        try:
            m_tokenizer, m_model = _attempt_load_mistral(local_only=False, bnb_config=bnb_config)
            mistral_loaded = True
            mistral_source = "download"
            print("✅ Mistral downloaded & loaded.")
        except Exception as e_dl:
            print(f"⚠️ Mistral failed to load (download attempt): {str(e_dl)[:400]}")
            mistral_loaded = False
else:
    print("No GPU detected — Mistral 4-bit quantized load skipped (GPU required).")
    mistral_loaded = False

# Force: require Mistral (no silent Flan fallback)
if not mistral_loaded:
    raise RuntimeError(
        "Mistral-7B did not load. This script requires Mistral to run (no fallback).\n\n"
        "Common fixes:\n"
        " - Install dependencies: pip install bitsandbytes sentencepiece\n"
        " - Ensure HF access (if gated model): run `huggingface-cli login` or set HF_TOKEN\n"
        " - Ensure sufficient free GPU VRAM (check `nvidia-smi`) and close other GPU tasks\n"
        " - If on Windows and bitsandbytes is problematic, consider WSL or Linux\n"
    )

# Wire final tokenizer/model
final_tokenizer = m_tokenizer
final_model = m_model
use_causal = True

# Model device
try:
    model_device = final_model.device
except Exception:
    model_device = next(final_model.parameters()).device

print(f"\n[5/6] System ready. Using Mistral (causal) (source: {mistral_source})")
print("Subsequent runs will reuse cached model files if present.")

# -------------------------
# Prompt & generation helpers (LLM-first)
# -------------------------
def make_final_prompt(user_query, evidence_summary):
    prompt = (
        "You are an admissions assistant. Answer the user's question using ONLY the facts "
        "present in the Evidence paragraph below. If the evidence does not contain the answer, "
        "respond: \"I don't know – please contact admissions@university.edu\".\n\n"
    )
    prompt += f"User question: {user_query}\n\n"
    prompt += "Evidence:\n" + evidence_summary + "\n\n"
    prompt += "If required, cite the snippet numbers (e.g., [Snippet 1]) and give the final answer in 2-4 sentences.\n\nAnswer:"
    return prompt

def summarize_snippets(snippets, max_chars=2000):
    combined = "\n".join([f"[Snippet {i+1}] {s}" for i, s in enumerate(snippets)])
    if len(combined) > max_chars:
        combined = combined[:max_chars - 3] + "..."
    return combined

def generate_final_answer(user_query, snippets, evidence_summary, max_new_tokens=150, num_beams=1):
    """
    Always use the loaded LLM to generate final answer.
    """
    if final_tokenizer is None or final_model is None:
        raise RuntimeError("LLM not loaded: final_tokenizer or final_model is None. Fix model loading.")

    prompt = make_final_prompt(user_query, evidence_summary)

    with torch.inference_mode():
        inputs = final_tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=2048 if use_causal else 1024)
        device_for_model = model_device
        inputs = {k: v.to(device_for_model) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            use_cache=True,
            pad_token_id=(final_tokenizer.eos_token_id if hasattr(final_tokenizer, "eos_token_id")
                          else final_tokenizer.pad_token_id)
        )
        # early_stopping only for encoder-decoder; skip for causal models (Mistral)
        if not use_causal:
            gen_kwargs["early_stopping"] = True

        gen = final_model.generate(**inputs, **gen_kwargs)
        out = final_tokenizer.decode(gen[0], skip_special_tokens=True)
        if "Answer:" in out:
            out = out.split("Answer:")[-1].strip()
        return out.strip()

def rag_answer_flow(user_query, topk_results, max_new_tokens_local=150, num_beams_local=1):
    snippets = [p for (_, _, p) in topk_results]
    evidence_summary = summarize_snippets(snippets, max_chars=2000)
    final_ans = generate_final_answer(user_query, snippets, evidence_summary,
                                      max_new_tokens=max_new_tokens_local,
                                      num_beams=num_beams_local)
    best_idx, best_score, best_snip = topk_results[0]
    agent_log = {
        "retrieval": [{"idx": int(idx), "score": float(score), "snippet": text}
                      for (idx, score, text) in topk_results],
        "evidence_summary": evidence_summary
    }
    return {
        "final_answer": final_ans,
        "best_snippet": {"idx": int(best_idx), "score": float(best_score), "snippet": best_snip},
        "agent_log": agent_log
    }

# -------------------------
# Quick test & comprehensive test
# -------------------------
print("\n" + "=" * 72)
print("QUICK TEST - FAISS CPU")
print("=" * 72)

q = "What is the eligibility for B.Tech?"
print(f"\nQuery: {q}")
print("Processing (FAISS on CPU)...")

t0 = time.time()
topk = retrieve_top_k(q, k=3)
res = rag_answer_flow(q, topk)
elapsed = time.time() - t0

print(f"\n⚡ FINAL ANSWER (generated in {elapsed:.2f}s):")
print(res["final_answer"])
print(f"\n📊 BEST SNIPPET (score: {res['best_snippet']['score']:.3f}):")
print(res["best_snippet"]["snippet"][:200] + "...")

# Comprehensive tests
def make_json_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    return obj

test_questions = [
    "What is the eligibility for B.Tech?",
    "What is the tuition fee per semester for B.Tech?",
    "Are there scholarships available?",
    "Is hostel accommodation available?",
    "How can I contact the admission office?",
    "What documents are required for admission?",
    "Does the university offer MBA programs?",
    "What is the eligibility for M.Tech admission?",
    "Does the university provide internship opportunities?",
    "What is the refund policy for admission withdrawal?"
]

print("\n" + "=" * 72)
print("RUNNING COMPREHENSIVE TESTS (10 questions) - FAISS CPU (LLM required)")
print("=" * 72)

results = []
total_start = time.time()

for i, question in enumerate(test_questions, 1):
    print(f"\n[{i}/10] {question}")
    t_start = time.time()
    try:
        topk = retrieve_top_k(question, k=3)
        output = rag_answer_flow(question, topk)
        elapsed_q = time.time() - t_start

        full_ans = output.get("final_answer") or ""
        print("  ✅ Answer:", full_ans)
        print(f"  ⚡ Time: {elapsed_q:.2f}s")

        results.append({
            "question": question,
            "final_answer": output.get("final_answer"),
            "best_snippet": output.get("best_snippet"),
            "retrieval": output.get("agent_log", {}).get("retrieval"),
            "summary": output.get("agent_log", {}).get("evidence_summary"),
            "time_sec": round(elapsed_q, 2)
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ❌ Error: {e}")
        results.append({"question": question, "error": str(e), "traceback": tb})

total_elapsed = time.time() - total_start

out_path = "rag_test_results_cpu_faiss.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(make_json_serializable(results), f, indent=4, ensure_ascii=False)

print("\n" + "=" * 72)
print("✅ TESTING COMPLETED (FAISS CPU with LLM)!")
print("=" * 72)
print(f"📁 Results saved to: {out_path}")
print(f"⚡ Total time: {total_elapsed:.2f}s")
print(f"📊 Average time per query: {total_elapsed / len(test_questions):.2f}s")
print("=" * 72)
