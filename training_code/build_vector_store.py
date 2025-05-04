import os
import pickle
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

# Streamlit multi-file uploader
uploaded_files = st.file_uploader("Upload PDF files to train", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_documents = []
    embedding_model = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(embedding_model)
    for uploaded_file in uploaded_files:
        # Save each file temporarily
        file_path = os.path.join("knowledge_base", uploaded_file.name)
        os.makedirs("knowledge_base", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load PDF
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        # Chunk
        documents = text_splitter.split_documents(docs)
        all_documents.extend(documents)  

        # Print structure
        for i, doc in enumerate(documents[:1]):  
            st.write(f"\n File: {uploaded_file.name}")
            st.write("Page Content:", doc.page_content[:300], "...")
            st.write("Metadata:", doc.metadata)

    # Print the number of documents (PDFs) and chunks
    st.write(f"Total number of uploaded PDF documents: {len(uploaded_files)}")
    st.write(f"Total number of chunks processed: {len(all_documents)}")
    
    # Create FAISS index from all collected documents
    # Path to index directory
    index_dir = "vector_db"
    
    # If we already have an index, load it; otherwise start fresh
    if os.path.exists(index_dir):
        existing_vs = FAISS.load_local(
            index_dir,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        initial_count = len(existing_vs.docstore._dict)
        st.write(f"Number of chunks BEFORE merge: {initial_count}")
        new_vs = FAISS.from_documents(all_documents, embedding_model)
        new_count = len(new_vs.docstore._dict)
        st.write(f"Number of NEW chunks to be merged: {new_count}")

    # Merge the new vectors into the existing index
        existing_vs.merge_from(new_vs)
        final_count = len(existing_vs.docstore._dict)
        st.write(f"Number of chunks AFTER merge: {final_count}")
        if final_count > initial_count:
            st.success("New documents successfully merged into existing index.")
        else:
            st.warning("No new documents merged. Check your input.")
        vector_store = existing_vs
    else:
        # No previous index, just create one
        vector_store = FAISS.from_documents(all_documents, embedding_model)

    # Save the merged (or newly created) index
    vector_store.save_local(index_dir)
    
    # Save all chunks
    with open("docs.pkl", "wb") as f:
        pickle.dump(all_documents, f)
    st.success("Training completed on all files.")
    st.success("FAISS index saved to `faiss_index/`.")
    st.success("Chunks saved to `docs.pkl`.")