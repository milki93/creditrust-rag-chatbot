# CrediTrust RAG Chatbot: Intelligent Complaint Analysis

CrediTrust Financial receives thousands of customer complaints each month across various financial products. This project builds an internal AI-powered chatbot using Retrieval-Augmented Generation (RAG) to help teams quickly analyze, summarize, and act on real customer complaints.

---

## Project Overview

The chatbot enables team members to ask natural language questions (e.g., “Why are customers unhappy with BNPL?”) and receive concise, context-aware answers based on real complaint narratives. The system leverages semantic search, text chunking, vector embeddings, and a vector database to provide fast and relevant insights.

---

## Data Preparation and Processing

- **Data Source:** CFPB Consumer Complaint Database ([link](https://www.consumerfinance.gov/data-research/consumer-complaints/))
- **Filtering:** Only complaints related to five key products are included: Credit card, Personal loan, Buy Now, Pay Later (BNPL), Savings account, and Money transfers.
- **Cleaning:** Narratives are lowercased, boilerplate phrases and artifacts (e.g., "see attached", "n/a") are removed, and personally identifiable information (PII) is masked. Domain-specific stopwords and very short or excessively long narratives are filtered out.

---

## Text Chunking, Embedding, and Indexing

- **Chunking:** Long complaint narratives are split into overlapping chunks (chunk size: 512, overlap: 100) to preserve context and improve retrieval accuracy.
- **Embedding:** Each chunk is embedded using the `sentence-transformers/all-MiniLM-L6-v2` model, chosen for its speed and strong semantic performance.
- **Vector Store:** Embeddings are indexed using FAISS, and metadata (complaint ID, product, etc.) is stored alongside each vector for traceability.
- **Persistence:** The vector store (`vector_store/complaints_faiss.index`) and metadata (`vector_store/complaints_metadata.pkl`) are saved for use in the chatbot pipeline.

---

## How to Run

1. **Preprocessing:**  
   Run the EDA and cleaning notebook or script

2. **Chunking & Embedding:**  
   Run the chunking and embedding script (`src/embedding.py` or the relevant notebook cells) to create the vector store in `vector_store/`.

---

## Project Rationale

- **Semantic Search:** Enables retrieval of the most relevant complaint narratives for any user query.
- **Chunking:** Ensures long narratives are broken into manageable, context-rich pieces for better embedding and retrieval.
- **Efficient Indexing:** FAISS provides fast, scalable similarity search across millions of complaint chunks.
- **Traceability:** Metadata ensures every retrieved chunk can be traced back to its original complaint and product.

---
