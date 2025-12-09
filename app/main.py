import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

ml_models = {}


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    source: str
    latency: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("1. Loading the embedding model...")
    ml_models["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding model {ml_models['embedding_model']} loaded")

    print("2. Connecting to Qdrant Client:")
    client = QdrantClient(url="http://localhost:6333")
    ml_models["qdrant_client"] = client
    print(f"Qdrant client connected to {client}")

    collection_name = "chat_cache"
    if not client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' not found. Creating...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # differs for each language model
                distance=Distance.COSINE,
            ),
        )
        print("Collection created!")
    else:
        print(f"Collection '{collection_name}' ready.")

    # control is yielded to the main application
    yield

    print("Cleaning up...")
    ml_models.clear()


app = FastAPI(
    title="Semantic Caching Layer",
    description="A caching layer for semantic embeddings",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    model_status = "loaded" if "embedding_model" in ml_models else "not loaded"
    return {"status": "ok", "model_status": model_status}


# 2. THE LOCAL INTELLIGENCE (Ollama Integration)
async def get_llm_response(prompt: str) -> str:
    # We talk to the local Ollama instance on port 11434
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "qwen2.5:0.5b",  # Make sure you pulled this model!
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()

            # Ollama returns a JSON object with a "response" field
            return response.json()["response"]

    except Exception as e:
        return f"Error contacting Ollama: {str(e)}"


@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()

    # This prevents the server from freezing while calculating vectors
    embedding_model = ml_models["embedding_model"]
    loop = asyncio.get_running_loop()

    vector = await loop.run_in_executor(
        None, partial(embedding_model.encode, request.prompt)
    )
    vector = vector.tolist()

    client = ml_models["qdrant_client"]

    # Search Cache
    search_results = client.query_points(
        collection_name="chat_cache",
        query=vector,
        limit=1,
    )

    threshold = 0.85

    # CACHE HIT
    if search_results.points and search_results.points[0].score > threshold:
        print(f"CACHE HIT! Score: {search_results.points[0].score}")
        return ChatResponse(
            response=search_results.points[0].payload["response"],
            source="cached",
            latency=time.time() - start_time,
        )

    # CACHE MISS
    print("CACHE MISS. Calling Local LLM (Ollama)...")
    llm_response = await get_llm_response(request.prompt)

    # Store in Cache
    client.upsert(
        collection_name="chat_cache",
        points=[
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload={"response": llm_response, "source": "llm"},
            )
        ],
    )

    return ChatResponse(
        response=llm_response, source="llm", latency=time.time() - start_time
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
