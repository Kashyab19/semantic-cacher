import asyncio
import random
import time

import httpx

# The questions we will ask. Some are repeated to test the cache.
# Notice "capital of france" appears twice.
QUERIES = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "Explain quantum computing in one sentence.",
    "What is the speed of light?",
    "Who wrote Hamlet?",
    "What is 2 + 2?",
    "What is the capital of France?",
    "What is the speed of light?",
]


async def send_query(client, prompt, query_id):
    print(f"Query {query_id}: Sending '{prompt}'...")
    start = time.time()

    try:
        response = await client.post(
            "http://localhost:8000/chat", json={"prompt": prompt}, timeout=30.0
        )
        response.raise_for_status()
        data = response.json()

        duration = time.time() - start
        source = data.get("source", "unknown")

        # Color code the output for swag
        # Green for Cache, Red for LLM
        color = "\033[92m" if source == "cached" else "\033[91m"
        reset = "\033[0m"

        print(
            f"{color}[{source.upper()}] Query {query_id} finished in {duration:.4f}s{reset}"
        )
        return {"id": query_id, "duration": duration, "source": source}

    except Exception as e:
        print(f"Query {query_id} failed: {e}")
        return None


async def main():
    print("--- STARTING SEMANTIC CACHE BENCHMARK ---")
    print(f"Firing {len(QUERIES)} requests sequentially...\n")

    results = []

    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(QUERIES):
            res = await send_query(client, prompt, i + 1)
            if res:
                results.append(res)

    # Calculate stats
    hits = [r for r in results if r["source"] == "cached"]
    misses = [r for r in results if r["source"] == "llm"]

    avg_hit = sum(r["duration"] for r in hits) / len(hits) if hits else 0
    avg_miss = sum(r["duration"] for r in misses) / len(misses) if misses else 0

    print("\n--- RESULTS ---")
    print(f"Total Requests: {len(QUERIES)}")
    print(f"Cache Hits:     {len(hits)}")
    print(f"Cache Misses:   {len(misses)}")
    print(f"Avg Miss Time:  {avg_miss:.4f}s (The 'Cost' of AI)")
    print(f"Avg Hit Time:   {avg_hit:.4f}s  (The 'Savings')")

    if avg_hit > 0:
        print(f"\nSpeedup Factor: {avg_miss / avg_hit:.1f}x faster!")


if __name__ == "__main__":
    asyncio.run(main())
