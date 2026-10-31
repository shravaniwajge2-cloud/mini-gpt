import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

#=====================================================================
# Page Config
#=====================================================================
st.title("C++ RAG with Ollama LLM")

#=====================================================================
# Load & Split Documents
#=====================================================================
@st.cache_resource
def load_vectorstore():
    # Make sure filename matches your workspace file ('cpp.txt')
    loader = TextLoader('cpp.txt', encoding="utf-8")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_docs = text_splitter.split_documents(documents)

    # Create embeddings and store them in a FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(final_docs, embeddings)

    # Fix: Return 'vectorstore' instead of undefined 'db'
    return vectorstore

#=====================================================================
# Load Vectorstore
#=====================================================================
db = load_vectorstore()

#=====================================================================
# Load Ollama LLM
#=====================================================================
llm = Ollama(model="gemma2:2b")

#=====================================================================
# Chat Interface
#=====================================================================
user_question = st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Thinking....."):
        # Retrieve relevant documents from the vectorstore
        docs = db.similarity_search(user_question, k=5)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based on the context below. If the question can't be answered based on the context, say "I don't know".

        Context:
            {context}
        Question:
             {user_question}
        Answer:"""

        # Generate a response using the LLM with the retrieved context
        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)
