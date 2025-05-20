import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "None"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / 'data' / 'index'
MODEL_DIR = BASE_DIR / 'models' / 'llama-3b-lora-merged'

faiss_index = faiss.read_index(str(INDEX_DIR / 'faqs.index'))
with open(INDEX_DIR / 'id_map.json', 'r', encoding='utf-8') as f:
    id_map = json.load(f)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

DISALLOWED_TOPICS = ["hack", "steal", "fraud", "terror", "illicit", "money laundering"]
PROFANITY = ["damn", "hell", "shit", "fuck"]

def is_disallowed(text: str) -> bool:
    text_lower = text.lower()
    if any(term in text_lower for term in DISALLOWED_TOPICS): return True
    if any(term in text_lower for term in PROFANITY): return True
    return False

def retrieve(query: str, top_k: int = 5):
    emb = embed_model.encode(query, convert_to_numpy=True).astype('float32')
    _, indices = faiss_index.search(emb.reshape(1, -1), top_k)
    return [id_map[idx] for idx in indices[0]]

def post_verify(answer: str, context: str) -> bool:
    ctx_emb = embed_model.encode(context, convert_to_numpy=True)
    ans_emb = embed_model.encode(answer, convert_to_numpy=True)
    cos_sim = np.dot(ctx_emb, ans_emb) / (np.linalg.norm(ctx_emb) * np.linalg.norm(ans_emb) + 1e-8)
    if cos_sim < 0.6: return False
    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(context.lower().split())
    if ans_tokens:
        overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
        if overlap < 0.2: return False
    return True

def generate_answer(query: str, top_k: int = 5) -> str:
    if is_disallowed(query):
        return "I’m sorry, I can’t assist with that."
    docs = retrieve(query, top_k)
    context = "".join(f"Q: {d['question']}\nA: {d['answer']}\n\n" for d in docs)
    instructions = (
        "You are a helpful customer-service assistant. "
        "Answer ONLY using the provided context passages. "
        "Do NOT hallucinate or add information not in context. "
        "If the answer is not contained in context, reply: 'I’m sorry, I don’t have enough information to answer that question.'"
    )
    prompt = f"{instructions}\n\nContext:\n{context}\nUser: {query}\nAssistant:"
    chat_response = client.chat.completions.create(
        model="./models/llama-3b-lora-merged",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answer queries only related to NUST. You will always refuse otherwise."},
            {"role": "user", "content": prompt},
        ]
    )
    generated = chat_response.choices[0].message.content
    answer = generated.replace(prompt, '').strip()
    if is_disallowed(answer) or not post_verify(answer, context):
        return "I’m sorry, I don’t have enough information to answer that question."
    return answer
