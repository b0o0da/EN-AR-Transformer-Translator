"""
EN-AR-Transformer-Translator — Streamlit Demo App
Loads the fine-tuned model from the Hugging Face Hub repo
"B0o0da/EN-AR-Model" (falls back to the base Helsinki-NLP model
if the Hub repo can't be loaded) and lets the user translate
English text into Arabic interactively.
"""

import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HUB_MODEL = "B0o0da/EN-AR-Model"           # repo_id only — NOT a blob/file URL
HUB_SUBFOLDER = "finetuned-opus-mt-en-ar"  # set to "" if your files sit in the repo root
BASE_MODEL = "Helsinki-NLP/opus-mt-en-ar"  # fallback if the Hub model can't be loaded
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
    # Try loading your fine-tuned model directly from the Hugging Face Hub.
    # This works automatically as long as the repo (B0o0da/EN-AR-Model)
    # contains the standard files: config.json, generation_config.json,
    # source.spm, special_tokens_map.json, target.spm, tf_model.h5 (or
    # pytorch_model.bin / model.safetensors), tokenizer_config.json, vocab.json
    try:
        kwargs = {"subfolder": HUB_SUBFOLDER} if HUB_SUBFOLDER else {}
        tokenizer = AutoTokenizer.from_pretrained(HUB_MODEL, **kwargs)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(HUB_MODEL, **kwargs)
        return tokenizer, model, HUB_MODEL, True
    except Exception as e:
        st.warning(f"Could not load `{HUB_MODEL}` from the Hub ({e}). Falling back to base model.")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = TFAutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        return tokenizer, model, BASE_MODEL, False


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
    st.caption(f"⚠️ Could not load `{HUB_MODEL}` — using base model `{loaded_model_name}`")

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