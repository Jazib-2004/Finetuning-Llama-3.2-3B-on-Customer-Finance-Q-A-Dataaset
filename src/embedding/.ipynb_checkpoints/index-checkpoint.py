import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
INDEX_DIR = BASE_DIR / 'data' / 'index'
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'all-MiniLM-L6-v2'

def load_processed(filename: str):
    with open(PROCESSED_DIR / filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_embeddings(texts: list, model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype='float32')

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatL2, path: Path):
    faiss.write_index(index, str(path))

def save_id_map(faqs: list, path: Path):
    id_map = []
    for idx, item in enumerate(faqs):
        id_map.append({
            'id': idx,
            'category': item['category'],
            'question': item['question'],
            'answer': item['answer'],
        })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

def main():
    faqs = load_processed('all_faqs_processed.json')
    texts = [f"{f['question']} {f['answer']}" for f in faqs]
    embeddings = generate_embeddings(texts)
    index = build_index(embeddings)
    save_index(index, INDEX_DIR / 'faqs.index')
    save_id_map(faqs, INDEX_DIR / 'id_map.json')
    print(f"Built index with {len(faqs)} vectors")
    
if __name__ == '__main__':
    main()
