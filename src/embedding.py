import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_vector_store(input_path: str, vector_store_path: str):
    """
    Create FAISS vector store from complaint narratives.
    
    Args:
        input_path: Path to the preprocessed complaints CSV
        vector_store_path: Directory to save vector store and metadata
    """
    try:
        # Load data
        logging.info("Loading data...")
        df = pd.read_csv(input_path)
        
        # Chunking parameters
        chunk_size = 256 # Reduced chunk size to mitigate sequence length warnings
        chunk_overlap = 50 # Adjusted chunk overlap accordingly
        logging.info(f"Using chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process chunks
        chunks = []
        metadata = []
        
        for idx, row in df.iterrows():
            text = row['cleaned_narrative']
            if pd.isna(text) or len(text.strip()) == 0:
                continue
                
            complaint_id = row.get('Complaint ID', idx)
            product = row['Product']
            
            try:
                text_chunks = splitter.split_text(text)
                for chunk in text_chunks:
                    chunks.append(chunk)
                    metadata.append({
                        'complaint_id': complaint_id,
                        'product': product,
                        'original_index': idx
                    })
            except Exception as e:
                logging.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        logging.info(f"Total chunks created: {len(chunks)}")
        
        # Embedding
        logging.info("Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=128
        )
        
        logging.info(f"Embeddings shape: {embeddings.shape}")
        
        # Create and save vector store
        logging.info("Creating vector store...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Ensure output directory exists
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Save index and metadata
        index_path = os.path.join(vector_store_path, 'complaints_faiss.index')
        metadata_path = os.path.join(vector_store_path, 'complaints_metadata.pkl')
        
        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info("Vector store and metadata saved successfully")
        
    except Exception as e:
        logging.error(f"Error in create_vector_store: {str(e)}")
        raise

if __name__ == "__main__":
    input_path = 'data/processed/filtered_complaints.csv'
    vector_store_path = 'vector_store'
    
    # Create vector store
    create_vector_store(input_path, vector_store_path)