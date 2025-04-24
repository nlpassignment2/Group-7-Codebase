import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set page config
st.set_page_config(page_title="Text Summarizer", layout="centered")

# Title
st.title("📝 Media Interview Transcript Summarizer")
st.markdown("This app uses `BART` to summarize long text or dialogue.")

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_path = "C:/Users/ccqer/Downloads/Bart_trained_mediasum"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Initialize session state for input and summary
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# Text input field bound to session state
st.session_state.input_text = st.text_area("Enter text to summarize", value=st.session_state.input_text, height=300)

# Summarize button
if st.button("Summarize"):
    if st.session_state.input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(
                st.session_state.input_text, return_tensors="pt", max_length=4096, truncation=True
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                summary_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    num_beams=4,
                    max_length=4096,
                    min_length=80,
                    length_penalty=1.0,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.session_state.summary_text = summary

# Show the summary if available
if st.session_state.summary_text:
    st.subheader("📌 Summary:")
    st.success(st.session_state.summary_text)
