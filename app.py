import streamlit as st
from sentence_transformers import SentenceTransformer

# Page configuration
st.set_page_config(page_title="Sentence Embeddings", page_icon="🤖")

st.title("🧠 Sentence Embedding Generator")

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# User input
text = st.text_area("Enter your text:", height=200)

if st.button("Generate Embedding"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)

        st.success("Embedding generated successfully!")

        # Show dimensions
        st.subheader("Embedding Information")
        st.write(f"**Dimension:** {len(embedding)}")

        # Show first few values
        st.subheader("First 20 Values")
        st.write(embedding[:20])

        # Optionally show the full embedding
        if st.checkbox("Show Full Embedding Vector"):
            st.write(embedding)
