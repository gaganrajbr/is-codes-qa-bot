from fastapi import FastAPI, Query
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()
DB_FILE = "is_codes.db"
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_db(query, top_k=3):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    query_emb = model.encode([query])[0]
    cursor.execute("SELECT id, content, embedding FROM codes")
    rows = cursor.fetchall()

    similarities = []
    for row in rows:
        emb = np.frombuffer(row[2], dtype=np.float32)
        score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        similarities.append((row[1], score))

    conn.close()
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

@app.get("/")
def home():
    return {"message": "IS Codes QA Bot is running 🚀"}

@app.get("/ask")
def ask(query: str = Query(..., description="Your question about IS codes")):
    results = search_db(query)
    return {"query": query, "answers": results}
