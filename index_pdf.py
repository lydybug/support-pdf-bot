import pdfplumber
import pickle
import re

pages = []

with pdfplumber.open("document.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            clean = re.sub(r"\s+", " ", text)
            pages.append({"page": i+1, "text": clean})

pickle.dump(pages, open("data.pkl", "wb"))

print("PDF processed successfully.")

