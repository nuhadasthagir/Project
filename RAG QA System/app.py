# app.py
# -*- coding: utf-8 -*-
"""
Streamlit RAG app (FAISS CPU retrieval, GPU embeddings/LLM optional)
- Retrieval: FAISS (CPU). Embeddings on GPU if available.
- Generation: prefer Mistral (cache-first), optional Flan fallback if Mistral fails and user forced Flan.
- UI: darker answer background for better visibility.
Run:
    streamlit run app.py
"""

import os
import time
import textwrap
import pickle
import traceback
import numpy as np
import torch
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)

# -------------------------
# Config (edit paths if needed)
# -------------------------
FAQ_PATH = "admission_faq.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE = "embeddings.npy"
PASSAGES_CACHE = "passages.pkl"
FAISS_INDEX_PATH = "faiss.index"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
FLAN_MODEL_ID = "google/flan-t5-large"

# Silence HF symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="University Admissions Assistant", page_icon="🎓", layout="wide")
st.title("🎓 University Admissions Assistant (RAG — FAISS CPU)")

# -------------------------
# Hardware detection
# -------------------------
gpu_available = torch.cuda.is_available()
device = "cuda" if gpu_available else "cpu"

# -------------------------
# Sidebar controls (Debug mode, force flan, retrieval knobs)
# -------------------------
with st.sidebar:
    st.header("⚙️ Run Options")
    debug_mode = st.checkbox("🧪 Debug Mode (skip LLM load)", value=True)
    force_flan = st.checkbox("⚠️ Force Flan-T5 (skip Mistral attempts)", value=False)
    st.markdown("---")

# Display system info
with st.sidebar:
    st.header("System")
    st.write("Device:", device)
    if gpu_available:
        try:
            st.write("GPU:", torch.cuda.get_device_name(0))
            st.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

# -------------------------
# Utility: cache housekeeping and FAQ-change detection
# -------------------------
def caches_exist():
    return os.path.exists(EMBED_CACHE) and os.path.exists(PASSAGES_CACHE) and os.path.exists(FAISS_INDEX_PATH)

def remove_caches():
    for p in (EMBED_CACHE, PASSAGES_CACHE, FAISS_INDEX_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def faq_newer_than_index():
    try:
        if not os.path.exists(FAQ_PATH):
            return False
        if not os.path.exists(FAISS_INDEX_PATH):
            return True
        return os.path.getmtime(FAQ_PATH) > os.path.getmtime(FAISS_INDEX_PATH)
    except Exception:
        return False

# Sidebar: Rebuild button
with st.sidebar:
    st.markdown("---")
    if st.button("🔁 Rebuild embeddings & FAISS (clear cache)"):
        remove_caches()
        st.experimental_rerun()

# Auto clear caches if FAQ changed
if faq_newer_than_index():
    remove_caches()

# -------------------------
# Helper: load & split passages
# -------------------------
@st.cache_resource
def load_and_split(path):
    if not os.path.exists(path):
        return None
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

# -------------------------
# Retrieval system: embeddings + FAISS index (cacheable)
# -------------------------
@st.cache_resource
def load_retrieval_system():
    passages = load_and_split(FAQ_PATH)
    if passages is None:
        raise FileNotFoundError(f"FAQ file not found: {FAQ_PATH} — create it and add passages separated by blank lines.")

    # SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)
    try:
        embed_model.to(device)
    except Exception:
        pass
    embed_model.eval()

    # Use cached artifacts if present and consistent
    if os.path.exists(EMBED_CACHE) and os.path.exists(PASSAGES_CACHE) and os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings = np.load(EMBED_CACHE)
            with open(PASSAGES_CACHE, "rb") as f:
                cached_passages = pickle.load(f)
            index = faiss.read_index(FAISS_INDEX_PATH)
            if len(cached_passages) == len(passages):
                return embed_model, index, passages, embeddings.shape[1]
        except Exception:
            pass

    # Otherwise compute embeddings (on device using convert_to_tensor)
    emb_tensor = embed_model.encode(
        passages,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=64,
        device=device
    )
    with torch.inference_mode():
        emb_tensor = emb_tensor / emb_tensor.norm(dim=1, keepdim=True).clamp(min=1e-9)
    embeddings = emb_tensor.cpu().numpy().astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    # persist
    np.save(EMBED_CACHE, embeddings)
    with open(PASSAGES_CACHE, "wb") as f:
        pickle.dump(passages, f)
    faiss.write_index(index, FAISS_INDEX_PATH)
    return embed_model, index, passages, d

# -------------------------
# LLM loader (cache-first) using BitsAndBytesConfig
# -------------------------
def safe_tokenizer_from_pretrained(model_id, local_only, hf_token):
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, token=hf_token, local_files_only=local_only)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True, token=hf_token, local_files_only=local_only)

@st.cache_resource
def load_language_model_try(mistral_ok=True):
    """
    Tries to load Mistral from local cache first (no network).
    If not present, tries to download (if allowed).
    Falls back to Flan (cache-first) only if mistral_ok=False or download fails.
    Returns (tokenizer, model, is_causal, model_label, source_tag)
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # BitsAndBytes config for quantized Mistral
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16"
    )

    # Helper to attempt mistral load
    def _load_mistral(local_only: bool):
        tokenizer = safe_tokenizer_from_pretrained(MISTRAL_MODEL_ID, local_only=local_only, hf_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=local_only
        )
        model.eval()
        return tokenizer, model

    # Try Mistral if allowed and GPU available
    if mistral_ok and gpu_available:
        # try load from cache first
        try:
            tok, mdl = _load_mistral(local_only=True)
            return tok, mdl, True, "Mistral-7B-Instruct", "mistral-cache"
        except Exception as e_cache:
            try:
                tok, mdl = _load_mistral(local_only=False)
                return tok, mdl, True, "Mistral-7B-Instruct", "mistral-download"
            except Exception:
                pass

    # Fallback: Flan-T5-Large (cache-first then download)
    try:
        tok = AutoTokenizer.from_pretrained(FLAN_MODEL_ID, local_files_only=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_ID,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        mdl.eval()
        return tok, mdl, False, "Flan-T5-Large", "flan-cache"
    except Exception:
        tok = AutoTokenizer.from_pretrained(FLAN_MODEL_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_ID,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        mdl.eval()
        return tok, mdl, False, "Flan-T5-Large", "flan-download"

# -------------------------
# Load retrieval system (will compute or load embeddings)
# -------------------------
with st.spinner("Loading retrieval system (FAISS CPU)..."):
    try:
        embed_model, index, passages, embed_dim = load_retrieval_system()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error("Failed to initialize retrieval system.")
        st.exception(e)
        st.stop()

# show initial info
with st.sidebar:
    st.header("Model & Retrieval")
    st.write("Passages:", len(passages))
    st.write("Embedding dim:", embed_dim)

# -------------------------
# LLM load (skip if debug mode)
# -------------------------
model_load_status = "Not loaded (debug mode)"
tokenizer = None
model = None
use_causal = False
model_label = "NONE"
model_source = "none"

if not debug_mode:
    with st.spinner("Loading language model (cache-first) — this may take a while..."):
        try:
            # If user forces Flan, skip Mistral attempt
            if force_flan:
                st.info("Force Flan-T5 enabled — skipping Mistral attempts.")
                tokenizer, model, use_causal, model_label, model_source = load_language_model_try(mistral_ok=False)
            else:
                tokenizer, model, use_causal, model_label, model_source = load_language_model_try(mistral_ok=True)
            model_load_status = f"{model_label} loaded ({model_source})"
            st.success(model_load_status)
        except Exception as e:
            model_load_status = f"Failed to load LLM: {str(e)[:200]}"
            st.warning(model_load_status)
            # as a fallback, try Flan
            try:
                tokenizer, model, use_causal, model_label, model_source = load_language_model_try(mistral_ok=False)
                model_load_status = f"{model_label} loaded ({model_source})"
                st.success(model_load_status)
            except Exception as final_e:
                st.error("Unable to load any LLM. Keep Debug Mode ON to use the UI or fix model weights.")
                st.exception(final_e)
else:
    st.warning("🧪 Debug Mode ON — LLM loading skipped. Uncheck Debug Mode in the sidebar to load the model.")
    model_load_status = "Skipped (debug mode)"
    model_label = "DEBUG-STUB"

# -------------------------
# Helper functions: retrieval + prompt + generation
# -------------------------
def embed_query(q):
    with torch.inference_mode():
        vec = embed_model.encode([q], convert_to_tensor=True, device=device)
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

def summarize_snippets(snippets, max_chars=5000):
    combined = "\n".join([f"[Snippet {i+1}] {s}" for i, s in enumerate(snippets)])
    if len(combined) > max_chars:
        combined = combined[:max_chars-3] + "..."
    return combined

def generate_final_answer(prompt_text, max_new_tokens_local=200, num_beams_local=1):
    if tokenizer is None or model is None:
        return "I don't know – please contact admissions@university.edu"
    with torch.inference_mode():
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048 if use_causal else 1024)
        device_for_model = next(model.parameters()).device
        inputs = {k: v.to(device_for_model) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens_local,
            do_sample=False,
            num_beams=num_beams_local,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id
        )
        # only add early_stopping for encoder-decoder models
        if not use_causal:
            gen_kwargs["early_stopping"] = True

        gen = model.generate(**inputs, **gen_kwargs)
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        if "Answer:" in out:
            out = out.split("Answer:")[-1].strip()
        return out.strip()

def rag_answer_flow(user_query, topk_results, max_new_tokens_local=200, num_beams_local=1):
    snippets = [p for (_, _, p) in topk_results]
    evidence_summary = summarize_snippets(snippets, max_chars=5000 if use_causal else 2500)
    # Always use LLM for final answer when loaded
    if tokenizer is None or model is None:
        # fallback: show retrieved evidence (useful in Debug Mode)
        best_snippets_text = "\n\n".join([f"[Snippet {i+1}] {s}" for i, s in enumerate(snippets)])
        agent_log = {
            "retrieval": [{"idx": int(idx), "score": float(score), "snippet": text}
                          for (idx, score, text) in topk_results],
            "evidence_summary": evidence_summary
        }
        return {
            "final_answer": f"(LLM unavailable) Evidence found:\n\n{best_snippets_text}",
            "best_snippet": {"idx": int(topk_results[0][0]), "score": float(topk_results[0][1]), "snippet": topk_results[0][2]},
            "agent_log": agent_log
        }

    prompt_text = make_final_prompt(user_query, evidence_summary)
    final_ans = generate_final_answer(prompt_text, max_new_tokens_local, num_beams_local)
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
# Streamlit UI (darker answer background)
# -------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: left; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.0rem; color: #666; text-align: left; margin-bottom: 1rem; }
    .snippet-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 0.5rem 0; }
    .score-badge { background-color: #1f77b4; color: white; padding: 0.2rem 0.5rem; border-radius: 0.3rem; font-size: 0.9rem; font-weight: bold; }
    /* DARKER ANSWER BOX */
    .answer-box { background-color: #15202b; color: #ffffff; padding: 1.0rem; border-radius: 0.5rem; border: 1px solid #0b2230; font-size: 1.0rem; margin: 1rem 0; }
    .answer-box a { color: #9ad3ff; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 University Admissions Assistant</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Model: <strong>{model_label}</strong> — Status: {model_load_status}</div>', unsafe_allow_html=True)

# Sidebar tuning
with st.sidebar:
    st.header("Retrieval & LLM Settings")
    k_value = st.slider("Number of snippets to retrieve (k)", min_value=1, max_value=5, value=3)
    max_new_tokens = st.slider("Max new tokens (answer length)", min_value=32, max_value=512, value=150, step=16)
    quality_speed = st.selectbox("Quality vs Speed", ["Balanced", "Faster (lower latency)", "Higher quality"])
    if quality_speed == "Faster (lower latency)":
        num_beams = 1
    elif quality_speed == "Higher quality":
        num_beams = 4
    else:
        num_beams = 1

    st.markdown("---")
    if st.button("Pre-warm model"):
        if tokenizer is None or model is None:
            st.warning("Model not loaded — uncheck Debug Mode and try again.")
        else:
            with st.spinner("Warming model..."):
                warm_prompt = "Say hello."
                try:
                    inputs = tokenizer(warm_prompt, return_tensors="pt", truncation=True).to(next(model.parameters()).device)
                    with torch.inference_mode():
                        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                    st.success("Model warmed.")
                except Exception as e:
                    st.error("Warm failed.")
                    st.exception(e)

# session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input("Ask your question:", value=st.session_state.query_input, placeholder="e.g., What are the admission requirements for B.Tech?")

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear History", use_container_width=True)

if clear_button:
    st.session_state.chat_history = []
    st.session_state.query_input = ""
    st.experimental_rerun()

if search_button and query:
    with st.spinner("Processing..."):
        start_time = time.time()
        try:
            topk = retrieve_top_k(query, k=k_value)
            result = rag_answer_flow(query, topk, max_new_tokens_local=max_new_tokens, num_beams_local=num_beams)
            elapsed = time.time() - start_time
            st.session_state.chat_history.append({
                "query": query,
                "result": result,
                "elapsed": round(elapsed, 2)
            })
            st.session_state.query_input = ""
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(traceback.format_exc())

# display history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Conversation History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"### Question {len(st.session_state.chat_history) - i}")
        st.markdown(f"**❓ {chat['query']}**")
        st.markdown(f'<div class="answer-box">✅ <strong>Answer:</strong><br>{chat["result"]["final_answer"]}</div>', unsafe_allow_html=True)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("⚡ Response Time", f"{chat['elapsed']:.2f}s")
        colB.metric("🎯 Best Score", f"{chat['result']['best_snippet']['score']:.3f}")
        colC.metric("📚 Snippets", k_value)
        speedup = 10 / chat['elapsed'] if chat['elapsed'] > 0 else 0
        colD.metric("🚀 Est. Speedup", f"~{speedup:.1f}x")

        with st.expander("📄 View Retrieved Snippets"):
            for j, ret in enumerate(chat["result"]["agent_log"]["retrieval"]):
                st.markdown(f'<div class="snippet-box">', unsafe_allow_html=True)
                st.markdown(f'**Snippet {j+1}** <span style="float:right" class="score-badge">Score: {ret["score"]:.3f}</span>', unsafe_allow_html=True)
                st.markdown(ret["snippet"])
                st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔍 View Evidence Summary"):
            st.text(chat["result"]["agent_log"]["evidence_summary"])

        st.markdown("---")
else:
    st.info("👋 Welcome! Ask a question to experience RAG (FAISS CPU). (Tip: enable LLM by unchecking Debug Mode)")

# footer
st.markdown("---")
st.caption(f"RAG | FAISS CPU retrieval | Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | LLM: {model_label} ({model_source})")
