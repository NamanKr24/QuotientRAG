import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import requests
import json
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    try:
        HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
    except AttributeError:
        st.error("Hugging Face API token not found. Please set it as an environment variable (HF_API_TOKEN) or in Streamlit secrets.")
        st.stop()

@st.cache_data
def load_dataframe():
    """Loads and preprocesses the DataFrame."""
    try:
        df = pd.read_csv("cleaned_quotes.csv")
        df['quote'] = df['quote'].astype(str)
        df['author'] = df['author'].astype(str)
        
        df['tags'] = df['tags'].apply(
            lambda x: [tag.strip() for tag in x.split(',')] if isinstance(x, str) and x else []
        )

        df.dropna(subset=['quote', 'author'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("cleaned_quotes.csv not found. Please run the main script to generate it.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or processing DataFrame: {e}")
        st.stop()


@st.cache_resource
def load_embedding_model_and_index():
    """Loads the fine-tuned Sentence Transformer model and FAISS index."""
    try:
        embed_model = SentenceTransformer('fine_tuned_miniLM_quotes')

        faiss_index = faiss.read_index("quotes_index.faiss")
        return embed_model, faiss_index
    except FileNotFoundError:
        st.error("Fine-tuned model ('fine_tuned_miniLM_quotes') or FAISS index ('quotes_index.faiss') not found. Please run the main script to generate them.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models or index: {e}")
        st.stop()

df = load_dataframe()
model, index = load_embedding_model_and_index()

def query_huggingface_llm(payload, model_id="HuggingFaceH4/zephyr-7b-beta"):
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        with st.spinner("Connecting to Hugging Face LLM..."):
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")
        st.write(f"Response content: {response.text}")
        if response.status_code == 503:
            st.warning("Model is currently loading on Hugging Face servers. This can take a minute. Please retry shortly.")
        return None
    except Exception as err:
        st.error(f"An unexpected error occurred while querying the LLM: {err}")
        return None

def retrieve_quotes(user_query, top_k=5):
    """Embeds the query and searches the FAISS index for relevant quotes."""
    query_embedding = model.encode([user_query])
    
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k) 

    results = []
    for idx, score in zip(indices[0], distances[0]):
        raw_tags = df.iloc[idx].get('tags', [])
        tags = raw_tags if isinstance(raw_tags, list) else []

        results.append({
            "quote": df.iloc[idx]['quote'],
            "author": df.iloc[idx]['author'],
            "tags": tags,
            "score": round(float(score), 4)
        })

    return results

def generate_response(user_query, retrieved_quotes):
    """Generates a response using the LLM based on retrieved quotes."""
    if not retrieved_quotes:
        return "I couldn't find any relevant quotes for your query. Please try rephrasing."

    context_quotes = "\n".join([
        f"- \"{q['quote']}\" (Author: {q['author']}, Tags: {', '.join(q['tags'])})"
        for q in retrieved_quotes
    ])

    prompt = f"""You are an intelligent assistant that provides insightful responses based on given quotes.
The user is looking for quotes related to: "{user_query}"

Here are some relevant quotes I found:
{context_quotes}

Based on the above quotes and the user's request, provide a concise and helpful answer. You can either directly present the most relevant quote(s), or synthesize information from them to answer the user's implicit question. If the quotes don't directly answer the query, explain that but still provide the most relevant ones.

Response:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 700,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }

    llm_response = query_huggingface_llm(payload)

    if llm_response and isinstance(llm_response, list) and len(llm_response) > 0:
        generated_text = llm_response[0].get('generated_text', '').strip()
        if generated_text.startswith(prompt.strip()):
            generated_text = generated_text[len(prompt.strip()):].strip()
        return generated_text
    else:
        return "I apologize, but I couldn't generate a coherent response at this moment. This might be due to API issues or the model loading on Hugging Face. Please try again or rephrase your query."


st.set_page_config(page_title="QuotientRAG", layout="wide")
st.title("📚 QuotientRAG")

st.markdown("""
This application retrieves relevant quotes based on your query and uses a Large Language Model (Zephyr-7b-beta) to generate a coherent answer.
The embedding model has been fine-tuned on a dataset of English quotes.
""")

user_query = st.text_input("Enter your query about quotes (e.g., 'quotes about insanity attributed to Einstein'):")

if st.button("Get Answer", type="primary"):
    if user_query:
        with st.spinner("Searching for quotes and generating response..."):
            retrieved_quotes = retrieve_quotes(user_query, top_k=5)

            st.subheader("🔍 Retrieved Quotes:")
            if retrieved_quotes:
                for q in retrieved_quotes:
                    st.markdown(f"**- \"{q['quote']}\"**")
                    st.write(f"  *Author: {q['author'].title()}, Tags: {', '.join([tag.title() for tag in q['tags']])}* [Similarity Score: `{q['score']}`]")
                    st.markdown("---")
            else:
                st.info("No relevant quotes found in the database for your query.")

            st.subheader("✨ Generated Response:")
            rag_response = generate_response(user_query, retrieved_quotes)
            st.write(rag_response)
    else:
        st.warning("Please enter a query to get started!")