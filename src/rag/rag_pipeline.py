import os
import sys
import logging
import gc
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..chatbot.retriever import ComplaintRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RAGPipeline:
    def __init__(self, vector_store_path: str):
        """
        Initialize the RAG pipeline with memory optimization
        Args:
            vector_store_path: Path to the vector store directory
        """
        try:
            self.retriever = ComplaintRetriever(vector_store_path)
            # Use CPU for summarization with a reliable model
            self.summarizer = pipeline("summarization", model="facebook/bart-base", device="cpu")
            # Use CPU for generation with a structured model
            self.llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-small", # Reverted to smaller model for faster iteration
                device="cpu",
                model_kwargs={
                    "max_length": 512,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_k": 50,
                    "top_p": 0.95
                }
            )
            # Initialize SentenceTransformer once for query embedding
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.embedding_model.max_seq_length = 128  # Align with retriever's max_seq_length
            
            logging.info("RAGPipeline initialized")
        except Exception as e:
            logging.error(f"Error initializing RAG pipeline: {str(e)}")
            raise

    def retrieve_relevant_chunks(self, query: str, k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks from the vector store with memory optimization
        Args:
            query: User's search query
            k: Number of chunks to retrieve
        Returns:
            List of retrieved chunks with metadata
        """
        try:
            # Embed the query
            query_embedding = self._embed_text(query)
            # Retrieve top-k chunks
            retrieved = self.retriever.retrieve(query_embedding, k)
            
            # Load the processed complaints data to get full text
            df = pd.read_csv('/home/milky/Documents/10 Academy/creditrust-rag-chatbot/data/processed/filtered_complaints.csv')
            
            # Get full text for retrieved complaints
            chunks = []
            for item in retrieved:
                # Get the full complaint text using the original_index
                original_idx = item['original_index']
                if original_idx < len(df):
                    complaint_text = df.loc[original_idx]['cleaned_narrative']
                    chunks.append({
                        'complaint_id': item['complaint_id'],
                        'product': item['product'],
                        'text': complaint_text
                    })
            
            # Clean up
            del df
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error in retrieve_relevant_chunks: {str(e)}")
            raise

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response based on retrieved chunks.
        Args:
            query: User's search query
            retrieved_chunks: List of retrieved chunks with metadata
        Returns:
            Generated response
        """
        try:
            # Format context from retrieved chunks
            context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
            
            # Create a structured prompt
            prompt = f"""Based on the following customer complaint excerpts, answer the question clearly and concisely. If the information is not in the excerpts, state that you don\'t have enough information. 

Context: {context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.llm(prompt)[0]['generated_text']
            
            # Clean up memory
            gc.collect()
            
            return response
        except Exception as e:
            logging.error(f"Error in generate_response: {str(e)}")
            return f"Error processing query: {str(e)}"
            raise

    def _embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using the specified model"""
        try:
            # Use the pre-initialized embedding model
            embedding = self.embedding_model.encode(text, show_progress_bar=False)
            # Convert to list of floats
            embedding_list = embedding.tolist()
            # No need to del model or gc.collect() here as model is reused
            return embedding_list
        except Exception as e:
            logging.error(f"Error in _embed_text: {str(e)}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the entire RAG pipeline
        Args:
            query: User's search query
        Returns:
            Dictionary containing retrieved chunks and generated response
        """
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieve_relevant_chunks(query)
            
            # Generate response
            response = self.generate_response(query, retrieved_chunks)
            
            return {
                'query': query,
                'retrieved_chunks': retrieved_chunks,
                'response': response
            }
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        vector_store_path = "/home/milky/Documents/10 Academy/creditrust-rag-chatbot/vector_store"
        
        # Initialize RAG pipeline
        rag = RAGPipeline(vector_store_path)
        
        # Evaluation Questions
        EVALUATION_QUESTIONS = [
            {
                "question": "What are common issues with credit card billing?",
                "expected_topics": ["billing errors", "late fees", "interest rates"],
                "expected_products": ["Credit card"],
                "expected_context": "customers experiencing billing issues"
            },
            {
                "question": "How do customers typically resolve disputes with their bank accounts?",
                "expected_topics": ["dispute resolution", "fraud", "unauthorized transactions"],
                "expected_products": ["Bank account"],
                "expected_context": "customers reporting unauthorized activity"
            },
            {
                "question": "What are the most frequent complaints about personal loans?",
                "expected_topics": ["interest rates", "loan terms", "payment issues"],
                "expected_products": ["Personal loan"],
                "expected_context": "customers discussing loan terms"
            },
            {
                "question": "How do customers handle unauthorized transactions?",
                "expected_topics": ["fraud", "dispute process", "reimbursement"],
                "expected_products": ["Credit card", "Bank account"],
                "expected_context": "customers reporting fraud"
            }
        ]

        def evaluate_system() -> str:
            """Run evaluation on predefined questions and return results as a Markdown table."""
            print("\nRunning Evaluation...")
            results = []
            
            for q in EVALUATION_QUESTIONS:
                print(f"\nEvaluating: {q['question']}")
                result = rag.process_query(q['question'])
                
                # Simple scoring logic (can be refined)
                quality_score = 3  # Default to 3 (neutral)
                comments = []

                # Check if response is empty or contains generic error
                if not result['response'] or "error" in result['response'].lower():
                    quality_score = 1
                    comments.append("Empty or error response.")
                else:
                    # Check for presence of expected topics in the response (simple keyword check)
                    found_expected_topic_in_response = False
                    for topic in q['expected_topics']:
                        if topic.lower() in result['response'].lower():
                            found_expected_topic_in_response = True
                            break
                    
                    # Check if retrieved chunks are relevant (simple keyword check)
                    retrieved_relevant_chunk = False
                    for chunk in result['retrieved_chunks']:
                        for topic in q['expected_topics']:
                            if topic.lower() in chunk['text'].lower():
                                retrieved_relevant_chunk = True
                                break
                        if retrieved_relevant_chunk: break

                    if found_expected_topic_in_response and retrieved_relevant_chunk:
                        quality_score = 5 # Good response and relevant retrieval
                    elif found_expected_topic_in_response or retrieved_relevant_chunk:
                        quality_score = 4 # Partially good (response has topic or retrieval is relevant)
                    else:
                        quality_score = 2 # Response/retrieval not clearly aligned
                
                # Add comments if response is short or seems off-topic
                if len(result['response'].split()) < 20 and quality_score > 2:
                    comments.append("Response is short, consider more detailed answer.")
                
                results.append({
                    "question": q['question'],
                    "generated_answer": result['response'],
                    "retrieved_sources": result['retrieved_chunks'][:2], # Show top 2 sources
                    "quality_score": quality_score,
                    "comments": "; ".join(comments) if comments else "Good."
                })
            
            # Generate Markdown table
            markdown_table = "### RAG Pipeline Evaluation Results\n\n"
            markdown_table += "| Question | Generated Answer | Retrieved Sources (Complaint ID, Product, Text Snippet) | Quality Score (1-5) | Comments/Analysis |\n"
            markdown_table += "|----------|------------------|-------------------------------------------------------|---------------------|-------------------|\n"

            for res in results:
                question = res['question']
                generated_answer = res['generated_answer'].replace("\n", " ")[:150] + "..." if len(res['generated_answer']) > 150 else res['generated_answer'].replace("\n", " ")
                
                sources_str = []
                for source in res['retrieved_sources']:
                    source_text_snippet = source['text'].replace("\n", " ")[:100] + "..." if len(source['text']) > 100 else source['text'].replace("\n", " ")
                    sources_str.append(f"ID: {source['complaint_id']}, Product: {source['product']}, Text: {source_text_snippet}")
                sources_formatted = "<br/>".join(sources_str)
                
                quality_score = res['quality_score']
                comments = res['comments']
                
                markdown_table += f"| {question} | {generated_answer} | {sources_formatted} | {quality_score} | {comments} |\n"
            
            print("\n" + markdown_table)
            return markdown_table

        markdown_output = evaluate_system()
        
        # Save the markdown output to README.md
        readme_path = "README.md"
        with open(readme_path, 'a') as f:
            f.write("\n" + markdown_output)
        print(f"Evaluation results appended to {readme_path}")

        # The interactive loop is temporarily removed for direct evaluation output
        # print("\nFinancial Complaint Analysis System")
        # print("Type 'quit' to exit")
        # print("Type 'eval' to run evaluation")
        
        # while True:
        #     try:
        #         query = input("\nYour question: ").strip()
        #         
        #         if query.lower() == 'quit':
        #             print("\nGoodbye!")
        #             break
        #
        #         if query.lower() == 'eval':
        #             evaluate_system()
        #             continue
        #
        #         # Process the query
        #         try:
        #             result = rag.process_query(query)
        #             print("\nAnswer:", result['response'])
        #             print("\nRetrieved Sources:")
        #             for i, chunk in enumerate(result['retrieved_chunks'][:2], 1):
        #                 print(f"\nSource {i}:")
        #                 print(f"Complaint ID: {chunk['complaint_id']}")
        #                 print(f"Product: {chunk['product']}")
        #                 print(f"Text: {chunk['text'][:200]}...")
        #         except Exception as e:
        #             print(f"Error processing query: {str(e)}")
        #             continue
        #
        #     except Exception as e:
        #         print(f"Unexpected error: {str(e)}")
        #         print("Please try again.")
        #         
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
        # print("The system could not be initialized. Please check the logs for more details.")
