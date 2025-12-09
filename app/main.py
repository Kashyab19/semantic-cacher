import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastembed import (  # one for sparse vectors and one for dense vectors
    SparseTextEmbedding,
    TextEmbedding,
)
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,  # <--- ADD THIS
    SparseVectorParams,
    VectorParams,
)
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
    print("1. Loading embedding models (FastEmbed)...")
    ml_models["dense_model"] = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    ml_models["sparse_model"] = SparseTextEmbedding(
        model_name="prithivida/Splade_PP_en_v1"
    )

    print("Models loaded.")

    print("2. Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)
    ml_models["qdrant"] = client

    # NEW COLLECTION NAME due to schema change
    collection_name = "chat_cache_v2"

    if not client.collection_exists(collection_name):
        print(
            f"Creating hybrid collection of Sparse and Dense vectors '{collection_name}'..."
        )
        client.create_collection(
            collection_name=collection_name,
            # DENSE vectors use VectorParams
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            # SPARSE vectors use SparseVectorParams
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,  # Keep in RAM for speed
                    )
                )
            },
        )
        print("Collection created!")

    yield
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


@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. generate hybrid vectors
    dense_model = ml_models["dense_model"]
    sparse_model = ml_models["sparse_model"]
    loop = asyncio.get_running_loop()

    def get_hybrid_embeddings(text):
        dense = list(dense_model.embed(text))[0]
        sparse_result = list(sparse_model.embed(text))[0]

        return dense, SparseVector(
            indices=sparse_result.indices.tolist(), values=sparse_result.values.tolist()
        )

    dense_vector, sparse_vector = await loop.run_in_executor(
        None, partial(get_hybrid_embeddings, request.prompt)
    )

    client = ml_models["qdrant"]

    # 2. SEARCH
    # We pass the raw dense_vector list and tell it to look in the "dense" index
    search_results = client.query_points(
        collection_name="chat_cache_v2",
        query=dense_vector,
        using="dense",
        limit=1,
        with_payload=True,
    )

    threshold = 0.85

    # 3. CACHE HIT CHECK (The Fix)
    if search_results.points and search_results.points[0].score > threshold:
        print(f"CACHE HIT! Score: {search_results.points[0].score}")
        cached_text = search_results.points[0].payload["response"]
        return StreamingResponse(streamer(cached_text), media_type="text/plain")

    # 4. CACHE MISS
    print("CACHE MISS. Streaming from LLM...")
    return StreamingResponse(
        stream_and_cache_generator(
            request.prompt,
            dense_vector,
            sparse_vector,
            client,
            ml_models,
        ),
        media_type="text/plain",
    )


# Helper functions:
async def streamer(text: str):
    """Yields the cached text in chunks to simulate a stream"""
    chunk_size = 10
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        await asyncio.sleep(0.01)  # Simulate latency


# Split 'vector' into 'dense_vector' and 'sparse_vector'
async def stream_and_cache_generator(
    prompt: str, dense_vector, sparse_vector, client, ml_models
):
    """
    1. Stream from Ollama
    2. Yield chunks to user
    3. Aggregate chunks
    4. Save to Qdrant at the end
    """
    full_response = ""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": True,  # <--- ENABLE STREAMING
    }

    try:
        async with httpx.AsyncClient() as http_client:
            async with http_client.stream(
                "POST", url, json=payload, timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json

                        try:
                            # Ollama sends JSON lines like {"response": "The", "done": false}
                            data = json.loads(line)
                            chunk = data.get("response", "")

                            if chunk:
                                full_response += chunk
                                yield chunk  # Send to user immediately!

                            if data.get("done", False):
                                # Stream is finished. Now we save to Cache!
                                # We need to run this in background or just do it here
                                print(
                                    f"Stream finished. Caching {len(full_response)} chars..."
                                )

                                client.upsert(
                                    collection_name="chat_cache_v2",
                                    points=[
                                        PointStruct(
                                            id=uuid.uuid4().hex,
                                            # FIX: Use the specific variables
                                            vector={
                                                "dense": dense_vector,  # <--- Updated
                                                "sparse": sparse_vector,  # <--- Updated
                                            },
                                            payload={
                                                "response": full_response,
                                                "source": "llm",
                                            },
                                        )
                                    ],
                                )
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"Error: {str(e)}"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
