import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import faiss, pickle
from sentence_transformers import SentenceTransformer
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("index.html")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("index.faiss")
chunks, metadata = pickle.load(open("data.pkl", "rb"))

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    q_vec = model.encode([q.question])
    _, ids = index.search(q_vec, 3)

    sources = [chunks[i] for i in ids[0]]
    pages = list(set(metadata[i]["page"] for i in ids[0]))

    prompt = f"""
Use ONLY the text below.
Give a short, clear summary (2–4 sentences).

Text:
{sources}

Question:
{q.question}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "pages": pages
    }
