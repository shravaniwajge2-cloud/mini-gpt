import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="College Knowledge Extractor", layout="wide")
st.title("🎓 College Knowledge Search (Pure Retrieval - No LLM)")

# Load documents and create local in-memory vector database using HuggingFace
@st.cache_resource
def build_vector_store():
    loader = DirectoryLoader("./docs", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    
    # Free, local embedding model (downloads automatically on first run)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    return vectorstore

with st.spinner("Loading document database..."):
    vectorstore = build_vector_store()

# Query UI
query = st.text_input("Enter your policy search query:")

if query:
    # Retrieve top matching source chunks directly (similarity search)
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    st.subheader("🔍 Matching Document Excerpts")
    
    if results:
        for idx, (doc, score) in enumerate(results):
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            st.markdown(f"### Result {idx+1} — *Source: `{source_file}`*")
            st.info(doc.page_content)
    else:
        st.warning("I cannot find the answer in the provided official documents.")