import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Upload & Load raw PDF(s)
pdfs_directory = 'pdfs/'

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

@st.cache_data(show_spinner="📄 Loading PDF...")
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

# Step 2: Create Chunks
@st.cache_data(show_spinner="✂️ Chunking text...")
def create_chunks(_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(_documents)
    return text_chunks

# Step 3: Setup Embeddings Model (Using Hugging Face Embeddings)
@st.cache_resource(show_spinner="🔗 Loading embedding model...")
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Step 4: Process and Save FAISS DB
def process_pdf(file_path: str, db_path: str):
    documents = load_pdf(file_path)
    chunks = create_chunks(documents)
    faiss_db = FAISS.from_documents(chunks, get_embedding_model())
    faiss_db.save_local(db_path)
