import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

st.set_page_config(page_title="GPT-2 Tokens & Embeddings")

st.title("GPT-2 Tokenizer and Embeddings")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter your text")

if st.button("Generate"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")

        token_ids = inputs["input_ids"][0].tolist()

        st.subheader("Token IDs")
        st.write(token_ids)

        st.subheader("Number of Tokens")
        st.write(len(token_ids))

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Token embeddings
        token_embeddings = outputs.last_hidden_state.squeeze(0)

        st.subheader("Embedding Shape")
        st.write(token_embeddings.shape)

        st.write(
            f"{token_embeddings.shape[0]} tokens × "
            f"{token_embeddings.shape[1]} embedding dimensions"
        )

        st.subheader("First Token Embedding")
        st.write(token_embeddings[0].tolist())

        if st.checkbox("Show All Token Embeddings"):
            st.write(token_embeddings.tolist())