import streamlit as st
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# Load BERT model and tokenizer (you can choose a specific pre-trained model)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a function to encode text using BERT
def encode_text(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True)['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Define a function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Define a function to search for papers based on user query using BERT embeddings
def search_papers(query, data, model, tokenizer):
    results = []
    query_embedding = encode_text(query, model, tokenizer)
    
    for index, row in data.iterrows():
        title = row['Title']
        abstract = row['Abstract']
        
        title_embedding = encode_text(title, model, tokenizer)
        abstract_embedding = encode_text(abstract, model, tokenizer)
        
        # Calculate the cosine similarity between the query and paper embeddings
        title_similarity = cosine_similarity(query_embedding, title_embedding)
        abstract_similarity = cosine_similarity(query_embedding, abstract_embedding)

        minimum_similarity = 0.5
        # You can adjust the threshold to control the matching sensitivity
        if title_similarity > minimum_similarity or abstract_similarity > minimum_similarity:
            results.append({
                'Title': title,
                'Authors': row['Authors'],
                'Abstract': abstract
            })
    
    return results

# Load your paper data from the CSV
df = pd.read_csv('2005.csv')
sample_data = df.sample(n=10, random_state=1)
data = sample_data  # Make sure to replace "your_paper_data.csv" with your actual data file

# Streamlit code for UI
st.title('Paper Search Engine')
user_query = st.text_input("Enter your search query: ")
if st.button('Search'):
    st.write('Search for ', user_query)
    search_results = search_papers(user_query, data, model, tokenizer)

    # Display the search results
    if not search_results:
        st.write("No matching papers found.")
    else:
        st.write(f"Found {len(search_results)} papers matching your query:\n")
        for result in search_results:
            st.write(f"Title: {result['Title']}")
            st.write(f"Authors: {result['Authors']}")
            st.write(f"Abstract: {result['Abstract']}\n")