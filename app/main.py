import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial

import httpx
from app.database import init_db, log_request
from app.security import get_current_user
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastembed import (
    SparseTextEmbedding,
    TextEmbedding,
)
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

ml_models = {}


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    source: str
    latency: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize SQLite Logger
    print("0. Initializing Traffic Log...")
    init_db()

    print("1. Loading embedding models (FastEmbed)...")
    ml_models["dense_model"] = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    ml_models["sparse_model"] = SparseTextEmbedding(
        model_name="prithivida/Splade_PP_en_v1"
    )

    print("Models loaded.")

    print("2. Connecting to Qdrant...")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    ml_models["qdrant"] = client

    # --- RETRY LOOP FOR DOCKER ---
    qdrant_ready = False
    for i in range(10):
        try:
            client.get_collections()
            qdrant_ready = True
            print(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
            break
        except Exception as e:
            print(f"Waiting for Qdrant to wake up... ({i + 1}/10)")
            time.sleep(1)

    if not qdrant_ready:
        raise RuntimeError("Could not connect to Qdrant after 10 seconds.")
    # ---------------------------

    collection_name = "chat_cache_v3"

    if not client.collection_exists(collection_name):
        print(f"Creating hybrid collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8, quantile=0.99, always_ram=True
                        )
                    ),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=1000,
            ),
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
    model_status = "loaded" if "dense_model" in ml_models else "not loaded"
    return {"status": "ok", "model_status": model_status}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    req_id = uuid.uuid4().hex
    start_time = time.time()
    tenant_id = user["tenant_id"]

    print(f"User ID from the tenant: {tenant_id}")
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
    search_results = client.query_points(
        collection_name="chat_cache_v3",
        query=dense_vector,
        using="dense",
        limit=1,
        with_payload=True,
        query_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
    )

    threshold = 0.85

    # 3. CACHE HIT CHECK
    if search_results.points and search_results.points[0].score > threshold:
        print(f"CACHE HIT! Score: {search_results.points[0].score}")
        point = search_results.points[0]
        cached_text = point.payload["response"]
        latency = time.time() - start_time

        log_request(req_id, request.prompt, "cache", latency, cached_text)
        return StreamingResponse(
            streamer(cached_text),
            media_type="text/plain",
            headers={"x-cache-id": point.id},
        )

    # 4. CACHE MISS
    print("CACHE MISS. Streaming from LLM...")
    return StreamingResponse(
        stream_and_cache_generator(
            request.prompt,
            dense_vector,
            sparse_vector,
            client,
            ml_models,
            req_id,
            start_time,
            tenant_id,
        ),
        media_type="text/plain",
        headers={"x-cache-id": req_id},
    )


# ... (imports)
from fastapi import HTTPException, status

# ... (inside app/main.py, before the helper functions)


@app.delete("/cache/{cache_id}")
async def delete_cache(cache_id: str, user: dict = Depends(get_current_user)):
    """
    Invalidates a specific cache entry.
    Security: Only deletes if the cache_id belongs to the requesting Tenant.
    """
    tenant_id = user["tenant_id"]
    client = ml_models["qdrant"]

    print(f"Request to delete cache {cache_id} from tenant {tenant_id}")

    # 1. Retrieve the point to check ownership
    # We must check if it exists AND if it belongs to this tenant
    points = client.retrieve(collection_name="chat_cache_v3", ids=[cache_id])

    if not points:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Cache entry not found"
        )

    point = points[0]

    # 2. Ownership Check (Critical Security)
    # If the payload 'tenant_id' doesn't match the requester, BLOCK IT.
    owner = point.payload.get("tenant_id")
    if owner != tenant_id:
        print(
            f"SECURITY ALERT: Tenant {tenant_id} tried to delete data owned by {owner}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not own this cache entry.",
        )

    # 3. Delete it
    client.delete(collection_name="chat_cache_v3", points_selector=[cache_id])

    print(f"Deleted cache {cache_id}")
    return {"status": "deleted", "id": cache_id}


# Helper functions:
async def streamer(text: str):
    """Yields the cached text in chunks to simulate a stream"""
    chunk_size = 10
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        await asyncio.sleep(0.01)


async def stream_and_cache_generator(
    prompt: str,
    dense_vector,
    sparse_vector,
    client,
    ml_models,
    req_id,
    start_time,
    tenant_id,
):
    full_response = ""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host}/api/generate"

    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient() as http_client:
            async with http_client.stream(
                "POST", url, json=payload, timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")

                            if chunk:
                                full_response += chunk
                                yield chunk

                            if data.get("done", False):
                                print(
                                    f"Stream finished. Caching {len(full_response)} chars..."
                                )

                                latency = time.time() - start_time

                                log_request(
                                    req_id, prompt, "llm", latency, full_response
                                )

                                client.upsert(
                                    collection_name="chat_cache_v3",
                                    points=[
                                        PointStruct(
                                            id=uuid.uuid4().hex,
                                            vector={
                                                "dense": dense_vector,
                                                "sparse": sparse_vector,
                                            },
                                            payload={
                                                "prompt": prompt,
                                                "response": full_response,
                                                "source": "llm",
                                                "tenant_id": tenant_id,
                                            },
                                        )
                                    ],
                                )
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"Error: {str(e)}"
