import gradio as gr
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.rag_pipeline import RAGPipeline

# Initialize the RAG Pipeline
VECTOR_STORE_PATH = "vector_store"
rag_pipeline = RAGPipeline(VECTOR_STORE_PATH)

def predict(message, history):
    try:
        # history is now a list of message dicts when type='messages'
        response_obj = rag_pipeline.process_query(message)
        answer = response_obj['response']
        retrieved_sources = response_obj['retrieved_chunks']

        # Prepare the current response for Gradio's messages format
        full_response_content = ""
        
        # Simulate streaming (optional but good for UX)
        for i in range(len(answer)):
            full_response_content = answer[:i+1]
            # Update the last assistant message in history
            yield history + [{'role': 'assistant', 'content': full_response_content}]

        sources_display = "\n\n**Retrieved Sources:**\n"
        if retrieved_sources:
            for i, chunk in enumerate(retrieved_sources, 1):
                source_text_snippet = chunk['text'].replace("\n", " ")[:200] + "..."
                sources_display += f"- **Source {i}**: {source_text_snippet}\n" # Modified to remove Product and Complaint ID
        else:
            sources_display += "No relevant sources found.\n"
            
        # Append sources to the final assistant message
        final_response_content = answer + sources_display
        yield history + [{'role': 'assistant', 'content': final_response_content}]

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        # Add error message to history as an assistant message
        yield history + [{'role': 'assistant', 'content': error_message}]


# Create the Gradio ChatInterface
demo = gr.ChatInterface(
    fn=predict,
    chatbot=gr.Chatbot(height=400, type='messages'), # Added type='messages' for modern Gradio handling
    title="CrediTrust Complaint Assistant",
    description="Ask questions about customer complaints and get AI-generated answers with sources.",
    examples=[
        ["What are common issues with credit card billing?"],
        ["How do customers typically resolve disputes with their bank accounts?"],
        ["What are the most frequent complaints about personal loans?"],
        ["How do customers handle unauthorized transactions?"]
    ],
)

if __name__ == "__main__":
    demo.launch(share=False) # Set share=True to get a public link (useful for sharing demos) 