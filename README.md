# 🌐 EN → AR Translator (Streamlit)

A simple Streamlit app that translates English text into Arabic using a fine-tuned
`Helsinki-NLP/opus-mt-en-ar` MarianMT model (TensorFlow).

## Project Structure

```
en-ar-translator/
├── app.py
├── requirements.txt
└── finetuned-opus-mt-en-ar/     # put your fine-tuned model files here
    ├── config.json
    ├── generation_config.json
    ├── source.spm
    ├── special_tokens_map.json
    ├── target.spm
    ├── tf_model.h5
    ├── tokenizer_config.json
    └── vocab.json
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Add your model

Copy your fine-tuned model's 8 files into the `finetuned-opus-mt-en-ar/` folder
(replacing the placeholder `.txt` file there). If you skip this step, the app
still runs — it just automatically falls back to the base
`Helsinki-NLP/opus-mt-en-ar` model from Hugging Face.

## Run

```bash
streamlit run app.py
```

Opens automatically at [http://localhost:8501](http://localhost:8501).

## Notes

- Model loading is cached with `@st.cache_resource`, so it only loads once per session.
- Translation uses beam search (`NUM_BEAMS = 4`) — lower it for faster (but slightly
  lower-quality) translations, or raise it for better quality at the cost of speed.
- `MAX_LEN` (100 tokens) caps both the input truncation and the generated output length.
