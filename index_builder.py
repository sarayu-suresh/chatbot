# index_builder.py
import os
import json
import faiss
import fitz
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")
data_dir = "data" 
chunks = []
chunk_sources = []

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_dir, filename)
        text = extract_text_from_pdf(file_path)
        doc_chunks = text.split("\n\n")
        chunks.extend(doc_chunks)
        chunk_sources.extend([filename] * len(doc_chunks))

embeddings = embedder.encode(chunks, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.idx")

with open("chunk_data.json", "w") as f:
    json.dump({"chunks": chunks, "sources": chunk_sources}, f)
