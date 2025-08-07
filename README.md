# CrediTrust Financial: AI-Powered Customer Complaint Assistant

This project develops an internal AI-powered chatbot for CrediTrust Financial, a fast-growing digital finance company. The tool transforms raw, unstructured customer complaint data into a strategic asset, enabling internal teams to quickly identify major complaint trends and understand customer pain points. It leverages Retrieval-Augmented Generation (RAG) to provide synthesized, evidence-backed answers to plain-English questions.

## ✨ Main Features

*   **Intelligent Complaint Answering:** Users can ask questions (e.g., "Why are people unhappy with BNPL?") and receive concise, AI-generated answers.
*   **Semantic Search:** Efficiently retrieves relevant complaint narratives using a FAISS vector database.
*   **Multi-Product Support:** Addresses inquiries across Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers.
*   **Interactive Chat Interface:** A user-friendly Gradio web interface for seamless interaction.
*   **Transparent Sourcing:** Displays the exact text chunks used by the AI to generate answers, enhancing trust and verifiability.

---

## 🛠️ Setup and Installation

To get the project up and running, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/milki93/creditrust-rag-chatbot.git
    cd creditrust-rag-chatbot
    ```
    *(If you are already in the project directory, skip this step)*

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to manually ensure `gradio`, `sentence-transformers`, `transformers`, `faiss-cpu`, and `pandas` are in your `requirements.txt` or install them directly if issues arise.)*

4.  **Place Raw Data:** Ensure your raw complaint data (`complaints.csv`) is located in `data/raw/`. If you are using a pre-filtered dataset, ensure `filtered_complaints.csv` is in `data/processed/`.

---

## 🚀 Core Components

### 1. Data Preparation (Preprocessing & Cleaning)

This component handles loading, cleaning, and filtering the raw CFPB complaint data to prepare it for the RAG pipeline.

*   **What it Does:**
    *   Filters data for the five target financial products.
    *   Removes records without consumer narratives.
    *   Cleans text by lowercasing, redacting PII, removing boilerplate, stopwords, and performing lemmatization.
    *   Filters narratives by length (5 to 500 words).
    *   Saves the processed data to `data/processed/filtered_complaints.csv`.

*   **Key Files:** `src/data_preprocessing.py`, `notebooks/01_eda_preprocessing.ipynb`

*   **Visualizations:**
    *   **![Distribution of Complaints by Product](images/product_distribution.png)**
        *Description: Bar chart showing the distribution of complaints across products after initial filtering (generated from `01_eda_preprocessing.ipynb`).*
    *   **![Distribution of Narrative Lengths](images/narrative_lengths.png)**
        *Description: Histogram illustrating the word count distribution of cleaned complaint narratives (generated from `01_eda_preprocessing.ipynb`).*

### 2. Text Embedding and Vector Store Indexing

This component converts cleaned narratives into numerical vector embeddings and indexes them for efficient semantic search.

*   **What it Does:**
    *   Reads `data/processed/filtered_complaints.csv`.
    *   Chunks narratives (`chunk_size=256`, `chunk_overlap=50`) for optimal embedding.
    *   Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
    *   Creates and saves a FAISS vector index (`complaints_faiss.index`) and associated metadata (`complaints_metadata.pkl`) to `vector_store/`.

*   **Key File:** `src/embedding.py`

*   **How to Build the Vector Store:**
    If you've updated `data/processed/filtered_complaints.csv` or changed chunking parameters, rebuild the vector store:
    ```bash
    source venv/bin/activate
    python3 src/embedding.py
    ```

### 3. RAG Core Logic and Evaluation

This is the core pipeline responsible for retrieving relevant information and generating AI responses.

*   **What it Does:**
    *   **Retrieval:** Embeds user queries and searches the FAISS vector store to retrieve the top 1 most relevant complaint chunk (`k=1`).
    *   **Generation:** Uses a refined prompt to combine the query and retrieved chunk, which is then fed to the `google/flan-t5-small` language model (LLM). It generates a concise answer, instructed to use only the provided context and state if information is insufficient.
    *   **Evaluation:** Includes an automated qualitative evaluation that runs predefined questions and appends results (answers, sources, quality scores) as a Markdown table to this `README.md`.

*   **Key Files:** `src/rag/rag_pipeline.py`, `src/chatbot/retriever.py`

*   **RAG Architecture:**
    **![High-Level RAG Architecture Diagram](images/rag_architecture.png)**
    *Description: A visual representation of the RAG pipeline's information flow, from user query to generated answer and retrieved sources. (You can generate this from the Mermaid code provided previously in the chat).*

*   **Qualitative Evaluation Results:**
    To run the RAG pipeline's internal evaluation and append the results to this `README.md`:
    ```bash
    source venv/bin/activate
    python3 -m src.rag.rag_pipeline
    ```
    **(The generated evaluation table will appear here after you run the above command.)**
    **![Evaluation Results Table](images/evaluation_table.png)**
    *Description: A screenshot of the qualitative evaluation results table, showing sample questions, AI-generated answers, retrieved sources, quality scores, and analysis comments. (You'll need to run the `rag_pipeline.py` to generate this section, then optionally screenshot it for the image).*

### 4. Interactive Chat Interface

This Gradio-based web application provides a user-friendly way to interact with the RAG chatbot.

*   **What it Does:**
    *   Presents a clean chat interface with an input box, submit button, and chat history display.
    *   Integrates seamlessly with the RAG pipeline to process user queries.
    *   Displays retrieved source text snippets below AI answers for transparency.
    *   Includes a basic streaming simulation for AI responses.

*   **Key File:** `app.py`

*   **How to Run the Chatbot Interface:**
    ```bash
    source venv/bin/activate
    python3 app.py
    ```
    Open the local URL (e.g., `http://127.0.0.1:7860`) displayed in your terminal in a web browser.

*   **Chatbot Visuals:**
    **![Screenshot of Chatbot Interface](images/chatbot_screenshot.png)**
    *Description: A screenshot of the running CrediTrust Complaint Assistant Gradio interface, showcasing a sample user query, the AI's response, and the retrieved sources.*

    **![GIF of Chatbot Interaction (Optional)](images/chatbot_interaction.gif)**
    *Description: An optional animated GIF demonstrating the interactive nature of the chatbot, including typing a query, receiving a streaming response, and viewing the attached sources.*

---

## ⚠️ Troubleshooting

*   **`FileNotFoundError: 'data/raw/complaints.csv'`:** Ensure `complaints.csv` is placed in the `data/raw/` directory.
*   **Memory/Crashing Issues:** If you experience memory issues during embedding or LLM inference:
    *   Ensure you have sufficient RAM.
    *   Confirm `src/data_preprocessing.py` uses a small `sample_size` (e.g., `20000`).
    *   Verify `src/embedding.py` uses `chunk_size=256` and `chunk_overlap=50`.
    *   Confirm `src/rag/rag_pipeline.py` uses `k=1` for retrieval and `google/flan-t5-small` as the LLM.
*   **`TypeError` or `AttributeError` in Gradio (`app.py`):** These were resolved by updating `app.py` to use `chatbot=gr.Chatbot(type='messages')` and adjusting the `predict` function's return format. Ensure your `app.py` matches the latest changes.
*   **"Token indices sequence length is longer..." warning:** This indicates the input to the LLM is too long. We mitigated this by reducing `k` to 1 and `chunk_size` to 256. If it persists, the model might require even shorter inputs or a different model.
*   **LLM Quality is Suboptimal:** The `google/flan-t5-small` model is a compromise for local execution. Its answers may be brief or less insightful. This is an acknowledged limitation due to resource constraints.

---

## 📈 Future Enhancements

*   **Improve LLM Answer Quality:**
    *   Explore fine-tuning `google/flan-t5-small` or other suitable smaller models on a domain-specific dataset.
    *   If hardware permits, consider larger, instruction-tuned models for richer responses.
*   **Refined Retrieval:** Implement multi-product filtering during retrieval to enhance precision for product-specific queries.
*   **Quantitative Evaluation:** Develop a comprehensive quantitative evaluation setup with a labeled test set and appropriate metrics.
*   **Deployment:** Containerize the application (Docker) and explore cloud deployment for scalability and accessibility.
*   **User Feedback:** Integrate a feedback mechanism in the UI to collect user ratings for continuous improvement.

---
