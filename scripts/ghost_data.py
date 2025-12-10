import time
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# CONFIG
COLLECTION_NAME = "chat_cache_v3"  # The old optimized collection
NUM_VECTORS = 10000  # How many ghosts we want
BATCH_SIZE = 500  # Commit every 500 records
VECTOR_SIZE = 384  # Must match your model (BGE-Small)

client = QdrantClient(host="localhost", port=6333)


def generate_batch(start_index, batch_size):
    """
    Generates random dense vectors.
    We skip sparse vectors for speed (Qdrant handles partial updates fine).
    """
    vectors = np.random.rand(batch_size, VECTOR_SIZE).astype(np.float32)

    points = []
    for i, vector in enumerate(vectors):
        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "dense": vector.tolist(),
                    # We omit 'sparse' to save generation time.
                    # The DB will accept dense-only points.
                },
                payload={
                    "prompt": f"Synthetic Stress Test Prompt {start_index + i}",
                    "response": "Synthetic data for load testing.",
                    "source": "synthetic_ghost",
                    "type": "stress_test",
                },
            )
        )
    return points


def run_stress_test():
    print(f"STARTING STRESS TEST: Seeding {NUM_VECTORS} vectors...")
    print(f"Target Collection: {COLLECTION_NAME}")

    start_time = time.time()
    total_uploaded = 0

    try:
        # Check if collection exists
        if not client.collection_exists(COLLECTION_NAME):
            print(
                f"❌ Collection {COLLECTION_NAME} not found! Run your app first to create it."
            )
            return

        # Loop in batches
        for i in range(0, NUM_VECTORS, BATCH_SIZE):
            batch = generate_batch(i, BATCH_SIZE)

            client.upsert(collection_name=COLLECTION_NAME, points=batch)

            total_uploaded += len(batch)
            print(
                f"   Batch {i // BATCH_SIZE + 1} Uploaded ({total_uploaded}/{NUM_VECTORS})"
            )

        duration = time.time() - start_time
        print(f"\n DONE! Inserted {NUM_VECTORS} vectors in {duration:.2f} seconds.")
        print(f" TPS (Transactions Per Second): {NUM_VECTORS / duration:.0f}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_stress_test()
