import os
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime
from dateutil.parser import parse
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import openai
from src.chatbot.retriever import ComplaintRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedComplaintChatbot:
    def __init__(self, vector_store_path: str, api_key: str = None):
        """
        Initialize the enhanced chatbot with optimized memory usage
        Args:
            vector_store_path: Path to the vector store directory
            api_key: OpenAI API key for enhanced features
        """
        self.retriever = ComplaintRetriever(vector_store_path)
        # Use a smaller model for better memory usage
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Initialize OpenAI client if API key provided
        self.use_openai = False
        if api_key:
            openai.api_key = api_key
            self.use_openai = True
            logging.info("OpenAI API initialized")
            
        logging.info("EnhancedComplaintChatbot initialized")

    def process_query(self, query: str, k: int = 3, date_range: tuple = None) -> Dict[str, Any]:
        """
        Process a user query with enhanced features
        Args:
            query: User's search query
            k: Number of results to return
            date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
        Returns:
            Dictionary containing query results and enhanced information
        """
        try:
            # Load the processed complaints data
            df = pd.read_csv('/home/milky/Documents/10 Academy/creditrust-rag-chatbot/data/processed/filtered_complaints.csv')
            
            # Apply date range filter if provided
            if date_range:
                start_date = parse(date_range[0])
                end_date = parse(date_range[1])
                df['Date received'] = pd.to_datetime(df['Date received'])
                df = df[(df['Date received'] >= start_date) & 
                       (df['Date received'] <= end_date)]
            
            # Only retrieve the top 1 result to reduce memory usage
            retrieved = self.retriever.retrieve(query, 1)
            
            # Get full text for retrieved complaints
            retrieved_texts = []
            for item in retrieved:
                # Get the full complaint text using the original_index
                original_idx = item['original_index']
                if original_idx < len(df):
                    complaint_text = df.iloc[original_idx]['cleaned_narrative']
                    # Generate summary
                    summary = self._summarize_text(complaint_text)
                    retrieved_texts.append({
                        'product': item['product'],
                        'complaint_id': item['complaint_id'],
                        'text': complaint_text,
                        'summary': summary
                    })
                else:
                    retrieved_texts.append({
                        'product': item['product'],
                        'complaint_id': item['complaint_id'],
                        'text': 'Complaint text not found',
                        'summary': 'No summary available'
                    })
            
            # Generate insights using OpenAI if available
            insights = self._generate_insights(query, retrieved_texts) if self.use_openai else None
            
            return {
                'query': query,
                'retrieved': retrieved,
                'retrieved_texts': retrieved_texts,
                'insights': insights
            }

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

    def _summarize_text(self, text: str) -> str:
        """Generate summary of a complaint text"""
        try:
            summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return "Summary generation failed"

    def _generate_insights(self, query: str, complaints: List[Dict[str, Any]]) -> str:
        """Generate insights about common issues using OpenAI"""
        try:
            # Prepare the prompt
            prompt = f"""Analyze the following complaints about {query} and provide insights about common issues:
            """
            
            for complaint in complaints:
                prompt += f"\nProduct: {complaint['product']}
Complaint: {complaint['text']}
Summary: {complaint['summary']}\n"
            
            prompt += "\nProvide key insights about common issues and patterns in these complaints."
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specialized in customer complaints."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating insights: {str(e)}")
            return "Insights generation failed"

    def chat(self):
        """Interactive chat loop with enhanced features"""
        print("\nEnhanced Complaint Chatbot")
        print("Type 'quit' to exit")
        print("Type 'help' for available commands")
        print("\nAvailable commands:")
        print("- Date range: 'date:2024-01-01 to 2024-12-31'")
        print("- Product filter: 'product:credit card'")
        print("- Summary only: 'summary'")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- Date range: 'date:2024-01-01 to 2024-12-31'")
                    print("- Product filter: 'product:credit card'")
                    print("- Summary only: 'summary'")
                    continue
                
                # Parse date range if provided
                date_range = None
                if 'date:' in query:
                    parts = query.split('date:')
                    query = parts[0].strip()
                    date_str = parts[1].strip()
                    if 'to' in date_str:
                        dates = date_str.split('to')
                        date_range = (dates[0].strip(), dates[1].strip())
                
                # Process query
                results = self.process_query(query, date_range=date_range)
                
                # Display results
                print("\nFound relevant complaints:")
                for i, item in enumerate(results['retrieved'], 1):
                    print(f"\nResult {i}")
                    print(f"Score: {item['score']:.4f}")
                    print(f"Product: {item['product']}")
                    print(f"Complaint ID: {item['complaint_id']}")
                    
                # Display complaint texts and summaries
                print("\nComplaint Details:")
                for i, complaint in enumerate(results['retrieved_texts'], 1):
                    print(f"\nComplaint {i} - {complaint['product']}")
                    print(f"Complaint ID: {complaint['complaint_id']}")
                    print("-" * 50)
                    print(f"Summary: {complaint['summary']}")
                    print("-" * 50)
                    print(f"Full text: {complaint['text']}")
                    print("-" * 50)
                    
                # Display insights if available
                if results['insights']:
                    print("\nInsights:")
                    print("-" * 50)
                    print(results['insights'])
                    print("-" * 50)
                    
            except Exception as e:
                print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    # Get OpenAI API key from environment variable or default to None
    api_key = os.getenv('OPENAI_API_KEY')
    
    vector_store_path = "/home/milky/Documents/10 Academy/creditrust-rag-chatbot/vector_store"
    chatbot = EnhancedComplaintChatbot(vector_store_path, api_key)
    chatbot.chat()
