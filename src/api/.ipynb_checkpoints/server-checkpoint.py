from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil

from ..retrieval.rag import generate_answer, retrieve
from ..ingestion.ingestion import main as ingest_main
from ..embedding.index import main as index_main


app = FastAPI(title="CS416 LLM Customer Service Assistant", version="1.1")

API_KEY = "None"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class ContextItem(BaseModel):
    category: str
    question: str
    answer: str

class RetrieveResponse(BaseModel):
    contexts: List[ContextItem]

@app.get("/", dependencies=[Depends(get_api_key)])
def read_root():
    return {"message": "CS416 LLM Assistant API is running."}

@app.get("/health", dependencies=[Depends(get_api_key)])
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(get_api_key)])
def chat(request: QueryRequest):
    try:
        answer = generate_answer(request.query, request.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(get_api_key)])
def retrieve_contexts(request: RetrieveRequest):
    try:
        docs = retrieve(request.query, request.top_k)
        contexts = [ContextItem(**doc) for doc in docs]
        return RetrieveResponse(contexts=contexts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/raw", dependencies=[Depends(get_api_key)])
async def upload_raw(file: UploadFile = File(...)):
    raw_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / file.filename
    try:
        with open(dest, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    try:
        ingest_main()
        index_main()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion/Indexing error: {e}")
    return {"filename": file.filename, "status": "processed"}
