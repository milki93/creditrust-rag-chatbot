import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import pickle

# Load a subset for prototyping
df = pd.read_csv('../data/processed/filtered_complaints.csv')
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

# Chunking
chunk_size = 512
chunk_overlap = 100
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

chunks = []
metadata = []
for idx, row in df.iterrows():
    text = row['cleaned_narrative']
    complaint_id = row.get('Complaint ID', idx)
    product = row['Product']
    for chunk in splitter.split_text(text):
        chunks.append(chunk)
        metadata.append({'complaint_id': complaint_id, 'product': product, 'original_index': idx})

print(f"Total chunks created: {len(chunks)}")

# Embedding (with larger batch size)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=128)
print(f"Embeddings shape: {embeddings.shape}")

# Indexing and saving
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs('../vector_store/', exist_ok=True)
faiss.write_index(index, '../vector_store/complaints_faiss.index')
with open('../vector_store/complaints_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Vector store and metadata saved in vector_store/")