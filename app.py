import os
import faiss
import numpy as np
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader
from sentence_transformers import SentenceTransformer
import fitz
import streamlit as st
from io import BytesIO
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pydub import AudioSegment
import io
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# LOGIN PAGE
# -------------------------------
def login_page():
    st.title("Quick Sum Login")
    tab = st.radio("Choose Option", ["Sign In", "Sign Up"])

    if tab == "Sign In":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign In"):
            st.session_state.logged_in = True
            st.success("Logged in successfully!")

    elif tab == "Sign Up":
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")
        if st.button("Sign Up"):
            if new_pass != confirm_pass:
                st.error("Passwords do not match!")
            else:
                st.success("User registered successfully!")

# -------------------------------
# QUICK SUM APP FUNCTION
# -------------------------------
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def split_into_chunks(text, max_length=512):
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def quick_sum_app():
    st.title("QUICK SUM")
    st.write("Upload a PDF, Excel file, or image to get a summary.")

    model = load_sentence_transformer()
    processor, blip_model = load_blip_model()
    embedding_dim = 384

    # Use session state to cache embeddings and index for the uploaded file
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
        st.session_state.metadata_store = []
        st.session_state.text_store = []

    def load_pdf(uploaded_file):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        metadata = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text
            metadata.append({"page_number": page_num + 1, "page_text": page_text})
        return text, metadata

    def load_excel(file_path):
        df = pd.read_excel(file_path)
        return "\n".join([" ".join(map(str, row)) for row in df.values])

    def embed_texts(texts):
        return np.array(model.encode(texts))

    def store_in_faiss(embeddings, metadata, texts):
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
        st.session_state.faiss_index = index
        st.session_state.metadata_store = metadata
        st.session_state.text_store = texts

    def retrieve(query, top_k=5):
        query_embedding = embed_texts([query])
        distances, indices = st.session_state.faiss_index.search(query_embedding, top_k)
        results = []
        for idx, i in enumerate(indices[0]):
            results.append((st.session_state.text_store[i], st.session_state.metadata_store[i], distances[0][idx]))
        return results

    uploaded_file = st.file_uploader("Choose a PDF or Excel file", type=["pdf", "xlsx"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            st.write("Processing PDF...")
            document_text, metadata = load_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            st.write("Processing Excel...")
            document_text = load_excel(uploaded_file)
            metadata = [{} for _ in range(len(document_text))]

        chunks = split_into_chunks(document_text)
        st.write(f"Document split into {len(chunks)} chunks.")
        embeddings = embed_texts(chunks)
        store_in_faiss(embeddings, metadata, chunks)
        st.write("Embeddings stored.")

    query = st.text_input("Enter a query for summary:")
    if query and st.session_state.faiss_index is not None:
        results = retrieve(query)
        relevant_text = " ".join([result[0] for result in results])

        # Mistral summary
        MISTRAL_API_KEY = "lHFGShbf91kbUx1vrJ2rqLDIaJnAhYBy"
        MISTRAL_API_URL = "https://codestral.mistral.ai/v1/fim/completions"
        if MISTRAL_API_KEY:
            try:
                payload = {
                    "model": "codestral-latest",
                    "prompt": f"Summarize based on user's query:\nQuery: {query}\nContent: {relevant_text}\nSummary:",
                    "max_tokens": 150,
                    "temperature": 0.7,
                }
                headers = {
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                }
                response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        summary = data["choices"][0]["message"]["content"].strip()
                        st.subheader("Summary:")
                        st.write(summary)
                    else:
                        st.error("API response missing 'choices'.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"API communication error: {e}")
        else:
            st.warning("Set MISTRAL_API_KEY.")

    # Image summary
    uploaded_image = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Generating summary...")
        inputs = processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.subheader("Image Summary:")
        st.write(caption)

# -------------------------------
# APP ENTRY POINT
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    quick_sum_app()
