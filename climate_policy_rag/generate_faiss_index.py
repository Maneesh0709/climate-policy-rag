import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Initialize the sentence transformer model for document embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to the documents
data_folder = './data'  # Ensure this is correct relative to where the script is

# Load the documents
documents = []
for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        with open(os.path.join(data_folder, file_name), 'r') as file:
            documents.append(file.read())
print("hi")
# Generate embeddings for all documents
document_embeddings = np.array([embedding_model.encode(doc) for doc in documents])

# Initialize FAISS index with L2 distance metric
dimension = document_embeddings.shape[1]  # Embedding dimension (e.g., 384 for 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(dimension)

# Add document embeddings to the FAISS index
index.add(document_embeddings)

# Save the FAISS index to the /src/ folder
faiss_index_path = './src/faiss_index.index'  # Adjust path if needed
faiss.write_index(index, faiss_index_path)

print(f"FAISS index saved at {faiss_index_path}")
