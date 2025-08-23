# create_db.py

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# ---------------------------
# SETTINGS
# ---------------------------
DATA_PATH = "data"        # folder where your IS Code PDFs are stored
CHROMA_PATH = "chroma_db" # folder to store the vector database

# ---------------------------
# Load documents
# ---------------------------
print("📂 Loading documents...")
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

print(f"✅ Loaded {len(documents)} documents")

# ---------------------------
# Split into chunks
# ---------------------------
print("✂️ Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"✅ Created {len(docs)} chunks")

# ---------------------------
# Create embeddings + store in Chroma
# ---------------------------
print("⚙️ Creating embeddings and storing in Chroma DB...")
embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
db.persist()

print(f"🎉 Database created at: {CHROMA_PATH}")
