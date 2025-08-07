import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
import logging
import gc
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ComplaintRetriever:
    def __init__(self, vector_store_path: str):
        """
        Initialize the retriever with memory optimization
        Args:
            vector_store_path: Path to the vector store directory
        """
        try:
            # Load FAISS index
            index_path = os.path.join(vector_store_path, 'complaints_faiss.index')
            if not os.path.exists(index_path):
                raise ValueError(f"Vector store not found at {index_path}")
            self.index = faiss.read_index(index_path)
            logging.info(f"Loaded FAISS index from {index_path}")
            
            # Load metadata
            metadata_path = os.path.join(vector_store_path, 'complaints_metadata.pkl')
            if not os.path.exists(metadata_path):
                raise ValueError(f"Metadata file not found at {metadata_path}")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logging.info(f"Loaded metadata from {metadata_path}")
            
            # Initialize embedding model with CPU and memory optimization
            # Force CPU usage for embedding model
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.model.max_seq_length = 128  # Reduce maximum sequence length
            
            # Pre-compute embeddings for common queries to save memory
            self._cache = {}
            self._cache_max_size = 100
            
        except Exception as e:
            logging.error(f"Error initializing retriever: {str(e)}")
            raise

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant complaints based on the query embedding
        Args:
            query_embedding: Embedding of the search query
            k: Number of results to return
        Returns:
            List of dictionaries containing retrieved items with metadata
        """
        try:
            # Search in FAISS index
            scores, indices = self.index.search(np.array([query_embedding]), k)
            
            # Get metadata for retrieved items
            retrieved = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    item = self.metadata[idx].copy()
                    item['score'] = float(score)
                    retrieved.append(item)
            
            return retrieved
        except Exception as e:
            logging.error(f"Error in retrieve: {str(e)}")
            raise
