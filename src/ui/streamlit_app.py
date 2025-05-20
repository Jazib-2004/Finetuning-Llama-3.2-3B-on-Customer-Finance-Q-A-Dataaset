import streamlit as st
from pathlib import Path
import shutil

from ..retrieval.rag import generate_answer, retrieve
from ..ingestion.ingestion import main as ingest_main
from ..embedding.index import main as index_main

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'

def get_contexts(query: str, top_k: int) -> str:
    contexts = retrieve(query, top_k)
    return "\n\n".join(f"Category: {d['category']}\nQ: {d['question']}\nA: {d['answer']}" for d in contexts)

def get_answer(query: str, top_k: int) -> str:
    return generate_answer(query, top_k)

def upload_and_index(uploaded_file) -> str:
    if uploaded_file is None:
        return "No file selected."
    try:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = DATA_RAW_DIR / Path(uploaded_file.name).name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        ingest_main()
        index_main()
        return f"File '{Path(uploaded_file.name).name}' uploaded and indexed successfully."
    except Exception as e:
        return f"Error during upload/index: {e}"

def main():
    st.set_page_config(page_title="NUST Bank LLM Assistant")
    st.title("NUST Bank Customer Service Assistant")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Chat", "Upload Data"])
    
    with tab1:
        query = st.text_area("Your Question", placeholder="Ask about bank services...", height=100)
        top_k = st.slider("Top-K Contexts", min_value=1, max_value=10, value=5, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retrieve Contexts"):
                if query:
                    contexts = get_contexts(query, top_k)
                    st.text_area("Retrieved Contexts", contexts, height=300)
                else:
                    st.warning("Please enter a question first.")
        
        with col2:
            if st.button("Get Answer"):
                if query:
                    answer = get_answer(query, top_k)
                    st.text_area("Assistant Response", answer, height=150)
                else:
                    st.warning("Please enter a question first.")
    
    with tab2:
        uploaded_file = st.file_uploader("Upload JSON or Excel", type=['json', 'xlsx'])
        if st.button("Upload & Index"):
            if uploaded_file:
                status = upload_and_index(uploaded_file)
                st.text_area("Upload Status", status, height=100)
            else:
                st.warning("Please select a file first.")

if __name__ == '__main__':
    main() 