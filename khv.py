from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Load the same embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Load the existing Chroma vector store
vector_store = Chroma(
    persist_directory="data",  # This is the folder storing your embeddings
    embedding_function=embedding_model
)

# Step 3: Define your query
query = "What are the hazardous waste storage rules for the paint industry?"

# Step 4: Perform semantic similarity search
results = vector_store.similarity_search(query, k=3)

# Step 5: Display results with metadata
for i, doc in enumerate(results, 1):
    print(f"\n🔹 Result {i}")
    print("-" * 40)
    print("Content:\n", doc.page_content.strip()[:500])
    print("\nMetadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    print("-" * 40)
