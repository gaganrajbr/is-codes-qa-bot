import os
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer

DB_FILE = "is_codes.db"
CODES_FOLDER = "codes"

def create_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS codes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        content TEXT,
        embedding BLOB
    )
    """)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    for file in os.listdir(CODES_FOLDER):
        if file.endswith(".txt") or file.endswith(".csv"):
            path = os.path.join(CODES_FOLDER, file)
            
            if file.endswith(".csv"):
                df = pd.read_csv(path)
                texts = df.astype(str).agg(" ".join, axis=1).tolist()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    texts = f.readlines()

            for text in texts:
                text = text.strip()
                if text:
                    embedding = model.encode([text])[0].tobytes()
                    cursor.execute("INSERT INTO codes (filename, content, embedding) VALUES (?, ?, ?)",
                                   (file, text, embedding))

    conn.commit()
    conn.close()
    print("✅ Database created successfully.")

if __name__ == "__main__":
    create_database()
