import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load data
df = pd.read_csv('data/data.csv')

# Preprocess data (modify as necessary)
df['text'] = df['speech_text'].apply(lambda x: x.lower().strip())

# Load models
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate embeddings
embeddings = embedding_model.encode(df['text'].tolist(), show_progress_bar=True)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

def retrieve(query, index, model, df):
    query_embedding = model.encode([query])[0]
    D, I = index.search(query_embedding.reshape(1, -1), k=5)
    results = [df.iloc[idx] for idx in I[0]]
    return results

def generate_response(query, retrieved_data, max_length=1024, max_new_tokens=100):
    retrieved_df = pd.DataFrame(retrieved_data)
    context = " ".join(retrieved_df['text'].tolist())
    
    # Encode query and context separately
    query_tokens = gpt2_tokenizer.encode(query, add_special_tokens=False)
    context_tokens = gpt2_tokenizer.encode(context, add_special_tokens=False)
    
    # Ensure the total length does not exceed the max_length limit
    total_length = len(query_tokens) + len(context_tokens) + 1  # +1 for the EOS token
    if total_length > max_length:
        context_tokens = context_tokens[:max_length - len(query_tokens) - 1]
    
    # Prepare input_ids
    input_ids = gpt2_tokenizer.encode(query + gpt2_tokenizer.eos_token) + context_tokens
    input_ids = torch.tensor([input_ids])
    
    # Generate response
    output = gpt2_model.generate(input_ids, max_new_tokens=max_new_tokens)
    response = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

st.title("RAG Application with Streamlit")

query = st.text_input("Enter your query:")
if query:
    results = retrieve(query, index, embedding_model, df)
    response = generate_response(query, results)
    st.write(response)
