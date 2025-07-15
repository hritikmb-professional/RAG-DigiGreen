import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="WasteGPT | RAG Assistant", layout="centered", initial_sidebar_state="auto")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #e8f5e9, #a5d6a7);
            font-family: 'Segoe UI', sans-serif;
        }

        .block-container {
            max-width: 800px;
            padding: 2rem;
            margin: auto;
            background-color: #ffffffdd;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        h1 {
            color: #000000;
            text-align: center;
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
        }

        .stTextInput > div > input {
            font-size: 16px;
            border-radius: 6px;
            border: 1px solid #ccc;
            padding: 0.6rem;
        }

        .stButton > button {
            background-color: #388e3c;
            color: white;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-size: 16px;
        }

        .stButton > button:hover {
            background-color: #2e7d32;
        }

        .response-box {
            margin-top: 20px;
            padding: 1.2rem;
            background-color: #dcedc8;
            border-left: 4px solid #388e3c;
            border-radius: 6px;
            font-size: 16px;
            line-height: 1.6;
            color: #000000;
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("WELCOME TO WASTEGPT — THE ONLY ADAPTIVE RAG ASSISTANT YOU NEED!")
st.markdown(
    """
    <div style='text-align: center; font-size: 1.1rem; margin-top: -1rem; color: #333;'>
        Ask any question related to hazardous or industrial waste. Powered by <strong>Gemini</strong>, <strong>LangChain</strong>, and real-time embeddings.
    </div>
    """, unsafe_allow_html=True
)

@st.cache_resource
def load_qa_pipeline():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="data", embedding_function=embedding_model)
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
    return qa_chain

qa_chain = load_qa_pipeline()

with st.form(key="question_form"):
    user_query = st.text_input("Enter your question:", placeholder="e.g., What are the penalties for improper disposal?")
    submit_button = st.form_submit_button("Get Answer")

if submit_button and user_query.strip():
    with st.spinner("Generating response..."):
        try:
            response = qa_chain.invoke({"query": user_query})
            st.markdown(f'<div class="response-box">{response["result"]}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
