import pdfplumber
import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = []
metadata = []

with pdfplumber.open("document.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            parts = text.split("\n\n")
            for part in parts:
                if len(part.strip()) > 60:
                    chunks.append(part)
                    metadata.append({"page": i + 1})

embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "index.faiss")
pickle.dump((chunks, metadata), open("data.pkl", "wb"))

print("PDF indexed successfully.")
