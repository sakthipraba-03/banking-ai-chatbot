# Banking AI Chatbot
An AI-powered assistant designed to handle SBI banking-related queries using a combination of intent classification and Retrieval-Augmented Generation (RAG). The system accurately identifies whether the query is banking-related and responds using information extracted from official banking documents.

# Features
- **Intent Detection** using a fine tuned BERT model to predict user query as banking or non banking.
- **RAG Pipeline** with FAISS vector and local LLM (Deepseek-r1) for answering banking questions.
- **Document-based Learning** uses real banking PDFs sourced from bank websites.
- User-friendly **Streamlit Web Interface** for interactive usage.

# Dataset Summary 
The dataset includes 5000 banking and 5000 non-banking queries, sourced from HuggingFace and public datasets. Non-banking queries span domain-specific, general-purpose, and casual interactions. After cleaning and merging as Query–Intent pairs, 100 ambiguous queries were added for data augmentation. A total of 10,100 queries were used to train the intent classification model for better generalization.

# How to Add Banking PDFs
- Run: streamlit run build_vector_store.py
- Redirect to the Streamlit UI.
- Upload your banking-related PDF files.
- The app will process, chunk, embed, and index them.

# How the System Works
1. **Document Upload & Vector DB Creation**
    - Banking PDFs are split semantically and embedded using HuggingFaceEmbedding models.
    - Faiss is used to create a searchable vector store and saved locally.
2. **RAG-Based Question Answering**
    - When a query is identified as banking-related, RAG retrieves relevant document chunks.
    - A local LLM (e.g., DeepSeek-r1 via Ollama) generates answers based on the retrieved context.
3. **End-to-End Inference**
    - User enters a query.
    - Intent model filters out non-banking questions.
    - If banking-related, the RAG system generates a concise answer in under two sentences.
    - If it's non-banking, you'll be asked to provide a banking query.

# Accuracy
Intent model Accuracy - 99%

# Install Dependencies
pip install -r requiremnts.txt

# Run the App
streamlit run main.py