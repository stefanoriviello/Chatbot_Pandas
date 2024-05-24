import streamlit as st
import os
import pickle
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatPremAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings import PremAIEmbeddings
from config import PREMAI_API_KEY
import create_embeddings

# Imposta la chiave API come variabile d'ambiente
os.environ["PREMAI_API_KEY"] = PREMAI_API_KEY

# Carica i chunk e le embeddings dal file pickle
with open('chunk_embeddings.pkl', 'rb') as f:
    all_chunks, chunk_embeddings = pickle.load(f)

# Inizializza l'embedder
model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=4316, model=model)

# Funzione per trovare i chunk più simili alla query
# Utilizza la similarità coseno per calcolare la similarità tra la query e i chunk, quindi restituisce i top_k chunk più simili con i loro valori di similarità associati 
def find_most_similar_chunks(query_embedding, chunk_embeddings, all_chunks, top_k=10):
    similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()
    most_similar_indices = similarities.argsort()[-top_k:][::-1]
    return [(all_chunks[i], similarities[i]) for i in most_similar_indices]


# Streamlit app
st.set_page_config(page_title="Chatbot Pandas 🐼✨", page_icon="✨")


st.markdown("""
<style>
    .stApp {
        background-color: ##1F1F1F;
        font-family: 'Arial', sans-serif;
    }
    .header-text {
        color: #4f8bcc;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subheader-text {
        color: #2e3b4e;
        font-size: 1.5rem;
    }
    .user-input {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .response-box {
        padding: 1.5rem;
        background-color: #4C5261;
        color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-text">Benvenuto nel Chatbot Pandas 🐼✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Fai una domanda e il chatbot ti risponderà!</div>', unsafe_allow_html=True)

# Input di testo per la query dell'utente
user_query = st.text_input("Inserisci la tua domanda:", key="user_input", help="Scrivi qui la tua domanda", label_visibility="collapsed")

if user_query:
    with st.spinner('mhhh... fammi pensare un attimo..'):
        # Embed della query utente
        query_embedding = embedder.embed_query(user_query)

        # Trova i chunk più simili
        most_similar_chunks = find_most_similar_chunks(query_embedding, chunk_embeddings, all_chunks)

        # Combina i chunk più simili per generare una risposta
        combined_text = " ".join([chunk for chunk, _ in most_similar_chunks])

        # Genera una risposta utilizzando i chunk più rilevanti
        chat = ChatPremAI(project_id=4316)
        system_message = SystemMessage(content=combined_text)
        human_message = HumanMessage(content=user_query)
        response = chat.invoke([system_message, human_message])

    # Visualizza la risposta
    st.markdown('<div class="subheader-text">Mi sembra di ricordare che...</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="response-box">{response.content}</div>', unsafe_allow_html=True)
