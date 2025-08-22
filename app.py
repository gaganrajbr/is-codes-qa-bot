# app.py
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# UI
st.title("📘 IS Codes Q&A Bot")
uploaded_files = st.file_uploader("Upload IS Codes PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    query = st.text_input("Ask a question about IS Codes")
    if query:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
        docs = retriever.get_relevant_documents(query)

        llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature":0.3, "max_length":512})
        context = "\n\n".join([d.page_content for d in docs])
        response = llm(f"Answer the following question based on IS Codes:\n\n{query}\n\nContext:\n{context}")
        st.write(response)
