import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import glob

st.title("📘 IS Codes Q&A Bot (Free LLM)")

# Load all PDFs from codes folder
code_files = glob.glob("codes/*.pdf")
docs = []
for file in code_files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Create embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load free HuggingFace LLM
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = st.text_input("Ask a question about IS Codes:")
if query:
    response = qa.run(query)
    st.write(response)
