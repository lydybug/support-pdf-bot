import os
import pickle
import re
import openai
from fastapi import FastAPI
from pydantic import BaseModel

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

pages = pickle.load(open("data.pkl", "rb"))

class Question(BaseModel):
    question: str

def score(text, words):
    return sum(text.lower().count(w) for w in words)

@app.post("/ask")
def ask(q: Question):
    words = re.findall(r"\w+", q.question.lower())
    ranked = sorted(pages, key=lambda p: score(p["text"], words), reverse=True)
    top = ranked[:2]

    source_text = " ".join(p["text"][:2000] for p in top)
    page_nums = [p["page"] for p in top]

    prompt = f"""
Use ONLY the text below.
Answer clearly in 2–4 sentences.

Text:
{source_text}

Question:
{q.question}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "pages": page_nums
    }

