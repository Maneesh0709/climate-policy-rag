from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = 'EleutherAI/gpt-neo-2.7B'  # Example: Free GPT model (can be replaced with any available model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load FAISS index
faiss_index_path = './src/faiss_index.index'  # Make sure this path is correct
index = faiss.read_index(faiss_index_path)

# Search function
def search_documents(query, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)
    return indices[0]

# Answer generation function
def generate_answer(query):
    # Retrieve the most relevant documents based on the query
    retrieved_docs = search_documents(query)

    # Read the content of the relevant documents
    context = ""
    data_folder = './data'  # Make sure this is the correct folder path for your documents
    data_files = [
        "carbon_price_brookings.txt", "climate_policy_africa.txt", "ipcc_adaptation_wg2.txt", 
        "ipcc_synthesis_spm.txt", "net_zero_critique.txt", "renewable_energy_brookings.txt", 
        "sr_renewable_ipcc.txt"
    ]
    
    # Only use the top relevant documents (limit context size)
    max_docs_to_use = 2  # Limit number of documents to use for context
    for i in retrieved_docs[:max_docs_to_use]:
        file_name = data_files[i]  # Map the retrieved index to the correct filename
        doc_path = os.path.join(data_folder, file_name)
        with open(doc_path, 'r') as file:
            context += file.read() + "\n"

    # Truncate the context to fit within the model's token limit
    max_token_length = 2048  # GPT-Neo 2.7B max token limit

    # Tokenize the question and context and truncate if necessary
    inputs = tokenizer(f"Question: {query}\nContext: {context}\nAnswer:", return_tensors="pt", truncation=True, max_length=max_token_length)

    # Check the input length before generating
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # If input length exceeds max length, truncate further (split the input into smaller chunks if necessary)
    if input_ids.shape[1] > max_token_length:
        print(f"Warning: Input exceeds model's max length, truncating to {max_token_length} tokens.")
        input_ids = input_ids[:, :max_token_length]
        attention_mask = attention_mask[:, :max_token_length]

    # Generate the response with max_new_tokens to avoid exceeding the model's max length
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150, num_return_sequences=1)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Simple command-line interaction
if __name__ == "__main__":
    print("Climate Change Policy Chatbot")
    while True:
        query = input("Ask a question about climate change policy: ")
        if query.lower() == 'exit':
            break
        answer = generate_answer(query)
        print(f"Answer: {answer}\n")
