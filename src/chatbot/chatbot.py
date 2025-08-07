import os
import pandas as pd
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
from retriever import ComplaintRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ComplaintChatbot:
    def __init__(self, vector_store_path: str):
        """
        Initialize the chatbot with the vector store path
        Args:
            vector_store_path: Path to the directory containing FAISS index and metadata
        """
        self.retriever = ComplaintRetriever(vector_store_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("ComplaintChatbot initialized")

    def process_query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Process a user query and return relevant complaints with full text
        Args:
            query: User's search query
            k: Number of results to return
        Returns:
            Dictionary containing query results and relevant complaints
        """
        try:
            # Load the processed complaints data
            df = pd.read_csv('/home/milky/Documents/10 Academy/creditrust-rag-chatbot/data/processed/filtered_complaints.csv')
            
            # Retrieve relevant complaints
            retrieved = self.retriever.retrieve(query, k)
            
            # Get full text for retrieved complaints
            retrieved_texts = []
            for item in retrieved:
                # Get the full complaint text using the original_index
                original_idx = item['original_index']
                if original_idx < len(df):
                    complaint_text = df.iloc[original_idx]['cleaned_narrative']
                    retrieved_texts.append({
                        'product': item['product'],
                        'complaint_id': item['complaint_id'],
                        'text': complaint_text
                    })
                else:
                    retrieved_texts.append({
                        'product': item['product'],
                        'complaint_id': item['complaint_id'],
                        'text': 'Complaint text not found'
                    })
            
            return {
                'query': query,
                'retrieved': retrieved,
                'retrieved_texts': retrieved_texts
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

    def chat(self):
        """Interactive chat loop"""
        print("\nComplaint Chatbot - Ask about financial complaints")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() == 'quit':
                print("\nGoodbye!")
                break
                
            try:
                results = self.process_query(query)
                print("\nFound relevant complaints:")
                for i, item in enumerate(results['retrieved'], 1):
                    print(f"\nResult {i}")
                    print(f"Score: {item['score']:.4f}")
                    print(f"Product: {item['product']}")
                    print(f"Complaint ID: {item['complaint_id']}")
                    
                # Print the full text of retrieved complaints
                print("\nComplaint Texts:")
                for i, complaint in enumerate(results['retrieved_texts'], 1):
                    print(f"\nComplaint {i} - {complaint['product']}")
                    print(f"Complaint ID: {complaint['complaint_id']}")
                    print("-" * 50)
                    print(complaint['text'])
                    print("-" * 50)
                    
            except Exception as e:
                print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    vector_store_path = "/home/milky/Documents/10 Academy/creditrust-rag-chatbot/vector_store"
    chatbot = ComplaintChatbot(vector_store_path)
    chatbot.chat()
