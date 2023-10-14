import streamlit as st
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Define a function to load BERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

# Load BERT model and tokenizer (you can choose a specific pre-trained model)
model_name = "bert-base-uncased"
model, tokenizer = load_model_and_tokenizer(model_name)

# Load the English stopwords
stop_words = set(stopwords.words('english'))

# Standard query preprocessing function
def preprocess_query(query):
    # Lowercase the query for consistency with the model
    query = query.lower()
    
    # Tokenize the query and remove stopwords
    tokens = query.split()
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove duplicate words
    unique_tokens = list(set(tokens))
    
    # Reconstruct the query
    query = ' '.join(unique_tokens)
    
    return query

# Define a function to encode text using BERT
@st.cache_data
def encode_text(text, _model, _tokenizer):
    input_ids = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)['input_ids']
    with torch.no_grad():
        outputs = _model(input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Define a function to calculate cosine similarity between two vectors
@st.cache_data
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Define a function to search for papers based on user query using BERT embeddings
def search_papers(query, data, model, tokenizer):
    results = []
    query = preprocess_query(query)
    query_embedding = encode_text(query, model, tokenizer)
    
    for index, row in data.iterrows():
        title = row['Title']
        abstract = row['Abstract']
        
        title_embedding = encode_text(title, model, tokenizer)
        abstract_embedding = encode_text(abstract, model, tokenizer)
        
        # Calculate the cosine similarity between the query and paper embeddings
        title_similarity = cosine_similarity(query_embedding, title_embedding)
        abstract_similarity = cosine_similarity(query_embedding, abstract_embedding)

        minimum_similarity = 0.6
        # You can adjust the threshold to control the matching sensitivity
        if title_similarity > minimum_similarity or abstract_similarity > minimum_similarity:
            results.append({
                'Title': title,
                'Authors': row['Authors'],
                'Abstract': abstract
            })
    
    return results

# Load your paper data from the CSV
@st.cache_data
def load_data():
    df = pd.read_csv('2005.csv')
    sample_data = df.sample(n=10, random_state=1)
    return sample_data

data = load_data()

# Streamlit code for UI
st.title('Paper Search Engine with Bert')
user_query = st.sidebar.text_input("Enter your search query: ")
if st.sidebar.button('Search'):
    with st.spinner('Searching for papers...'):
        search_results = search_papers(user_query, data, model, tokenizer)

    # Display the search results
    if not search_results:
        st.error("No matching papers found.")
    else:
        st.success(f"Found {len(search_results)} papers matching your query:")
        for result in search_results:
            st.markdown(f"## {result['Title']}")
            st.markdown(f"**Authors:** {result['Authors']}")
            st.markdown(f"**Abstract:** {result['Abstract']}\n")
