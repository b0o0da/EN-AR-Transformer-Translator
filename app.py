"""
EN-AR-Transformer-Translator — Streamlit Demo App
Loads the fine-tuned (or base) Helsinki-NLP/opus-mt-en-ar model and
lets the user translate English text into Arabic interactively.
"""

import os

import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "finetuned-opus-mt-en-ar"     # folder with your fine-tuned model files
BASE_MODEL = "Helsinki-NLP/opus-mt-en-ar"  # fallback if no fine-tuned model found
MAX_LEN = 100
NUM_BEAMS = 4  # beam search width — higher = better quality, slower generation

st.set_page_config(
    page_title="EN → AR Translator",
    page_icon="🌐",
    layout="centered",
)


# ------------------------------------------------------------------
# Load model & tokenizer (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    # A fine-tuned MarianMT export has these files in MODEL_PATH:
    # config.json, generation_config.json, source.spm, special_tokens_map.json,
    # target.spm, tf_model.h5, tokenizer_config.json, vocab.json
    required_files = ["config.json", "tf_model.h5", "vocab.json"]
    has_finetuned = os.path.isdir(MODEL_PATH) and all(
        os.path.isfile(os.path.join(MODEL_PATH, f)) for f in required_files
    )

    model_name = MODEL_PATH if has_finetuned else BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model, model_name, has_finetuned


tokenizer, model, loaded_model_name, using_finetuned = load_model()


def translate(text: str) -> str:
    if not text.strip():
        return ""
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        max_length=MAX_LEN,
    )
    output_ids = model.generate(
        **inputs,
        max_length=MAX_LEN,
        num_beams=NUM_BEAMS,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🌐 EN → AR Translator")

if using_finetuned:
    st.caption(f"Powered by your fine-tuned model · `{loaded_model_name}`")
else:
    st.caption(f"⚠️ Fine-tuned model not found in `{MODEL_PATH}/` — using base model `{loaded_model_name}`")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("English")
    source_text = st.text_area(
        "Enter text to translate",
        height=200,
        value=st.session_state.get("example_text", ""),
        placeholder="Type an English sentence here...",
        label_visibility="collapsed",
    )
    translate_clicked = st.button("Translate →", type="primary", use_container_width=True)

with col2:
    st.subheader("Arabic")
    output_placeholder = st.empty()
    output_placeholder.markdown(
        "<div style='height:200px; border:1px solid #444; border-radius:8px; "
        "padding:12px; direction:rtl; text-align:right; color:#aaa;'>"
        "Translation will appear here...</div>",
        unsafe_allow_html=True,
    )

if translate_clicked:
    if not source_text.strip():
        st.warning("Please enter an English sentence to translate.")
    else:
        with st.spinner("Translating..."):
            translated = translate(source_text)
        output_placeholder.markdown(
            f"<div style='height:200px; border:1px solid #444; border-radius:8px; "
            f"padding:12px; direction:rtl; text-align:right; font-size:18px; overflow-y:auto;'>"
            f"{translated}</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ------------------------------------------------------------------
# Example sentences
# ------------------------------------------------------------------
st.caption("Try an example:")
examples = [
    "Hi, how are you today?",
    "I love learning about artificial intelligence.",
    "This movie was absolutely amazing.",
]

example_cols = st.columns(len(examples))
for col, example in zip(example_cols, examples):
    with col:
        if st.button(example, use_container_width=True):
            st.session_state["example_text"] = example
            st.rerun()
