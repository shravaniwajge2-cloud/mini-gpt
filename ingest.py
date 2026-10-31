import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db():
    # Ensure the docs directory exists
    if not os.path.exists("docs"):
        os.makedirs("docs")
        print("Created 'docs' folder. Please place your PDF documents there and re-run.")
        return

    print("Loading documents from 'docs/'...")
    loader = PyPDFDirectoryLoader("docs")
    raw_documents = loader.load()
    
    if not raw_documents:
        print("No documents found in the 'docs' folder! Please add at least one PDF.")
        return

    # Chunking strategy: 800 characters per chunk with 100 overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split {len(raw_documents)} pages into {len(chunks)} chunks.")

    # 100% Local Embeddings (No API Key Required)
    print("Generating local embeddings using HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Save to local Chroma database
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Vector database successfully built and saved to './chroma_db'!")

if __name__ == "__main__":
    build_vector_db()