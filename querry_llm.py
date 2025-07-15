import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="data",
    embedding_function=embedding_model
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_output_tokens=512
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

print("Welcome to the Hazardous Waste Adaptive RAG Assistant!")
print("Hazardous Waste RAG Assistant (Ask a question or type 'exit')")

while True:
    user_input = input("Your question: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        response = qa_chain.invoke({"query": user_input})
        print(f"\nAnswer:\n{response['result']}")
    except Exception as e:
        print(f"Error: {str(e)}")
