# app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="IS Codes QA Bot", layout="wide")
st.title("📘 IS Codes QA Bot (LangChain + Chroma)")

# ---------------------------
# Setup embeddings + DB
# ---------------------------
# You need to set OPENAI_API_KEY in your environment
embeddings = OpenAIEmbeddings()

# Assume you already have a Chroma DB directory called "chroma_db"
# If not, you’ll need to create it by loading and embedding your docs first
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# Prompt + QA chain
# ---------------------------
prompt_template = """
You are an assistant for question-answering tasks about IS Codes.
Use the following context to answer the question.
If you don’t know the answer, just say you don’t know. Do not make up answers.

Context:
{context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
)

# ---------------------------
# Streamlit Input
# ---------------------------
query = st.text_input("Ask a question about IS Codes:")

if query:
    with st.spinner("Searching IS Codes..."):
        response = qa_chain.run(query)
    st.markdown("### Answer")
    st.write(response)
