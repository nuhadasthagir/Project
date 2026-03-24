# st_test.py
import streamlit as st
st.set_page_config(page_title="Streamlit Test")
st.title("Streamlit Render Test ✅")
st.write("If you see this, Streamlit UI is rendering correctly.")
if st.button("Say hello"):
    st.write("Hello from Streamlit!")