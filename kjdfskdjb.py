from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

loader = PyPDFLoader("data_pdf.pdf")  # Replace with your PDF filename
documents = loader.load()

#Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

#Set up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Create embeddings (optional – useful for debugging)
try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    print("Vector embeddings created successfully.")
except Exception as e:
    print(f" Error creating vector embeddings: {e}")

#Create vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")
vector_store.add_documents(documents=texts)
vector_store.persist()  # Ensure data is saved

#Run a test similarity query
try:
    test_query = "Iam an organic chemical manufacturer,what ph should i maintain "
    results = vector_store.similarity_search(query=test_query, k=5)

    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    final_results = list(unique_results.values())[:3]
    print("Top 3 search results:\n")
    for i, doc in enumerate(final_results, 1):
        print(f"Result {i}:")
        print(doc.page_content[:300], "\n---\n")
except Exception as e:
    print(f"Error during test query: {e}")
