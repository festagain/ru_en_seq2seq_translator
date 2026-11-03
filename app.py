from pathlib import Path

import streamlit as st
import torch

from src.inference import load_model, translate_sentence
from src.preprocessing import clean_text


CHECKPOINT_PATH = "models/best_bpe_attention.pt"
RU_MODEL_PATH = "artifacts/tokenizers/ru_bpe.model"
EN_MODEL_PATH = "artifacts/tokenizers/en_bpe.model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_translator():
    return load_model(
        name="bpe_attention_lstm",
        checkpoint_path=CHECKPOINT_PATH,
        ru_model_path=RU_MODEL_PATH,
        en_model_path=EN_MODEL_PATH,
        device=DEVICE
    )


st.set_page_config(page_title="RU → EN | Переводчик", page_icon="🌐")
st.title("Русский → English")
st.caption("Перевод коротких предложений с русского на английский.")

with st.form("translation_form"):
    sentence = st.text_area(
        "Текст на русском",
        placeholder="Мне пора идти спать.",
        height=140,
        max_chars=1000
    )
    submitted = st.form_submit_button("Перевести", type="primary")

if submitted:
    sentence = clean_text(sentence)

    if not sentence:
        st.warning("Введите текст для перевода.")
    else:
        if len(sentence.split()) > 30:
            st.info("Модель обучалась на коротких предложениях. Длинный текст лучше разделить.")

        try:
            with st.spinner("Перевожу…"):
                bundle = load_translator()
                translation = translate_sentence(sentence, bundle, max_length=50)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Не удалось выполнить перевод: {error}")
        else:
            st.subheader("Перевод")
            if translation:
                st.text(translation)
            else:
                st.info("Модель вернула пустой перевод. Попробуйте другую формулировку.")
