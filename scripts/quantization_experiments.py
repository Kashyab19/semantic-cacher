import time

import numpy as np

# configs:
DIMENSIONS = 384
NUM_VECTORS = 10000
TOP_K = 10


def generate_vectors(count, dim):
    """Generates random normalized vectors (simulating embeddings)."""
    # 1. Create random noise
    vecs = np.random.randn(count, dim).astype(np.float32)
    # 2. Normalize
    # We force reshape to (-1, 1) to ensure it becomes a column vector (N, 1)
    norms = np.linalg.norm(vecs, axis=1).reshape(-1, 1)

    return vecs / norms


def simulate_quantization(vectors):
    """
    Simulate quantization process.

    Args:
        vectors (np.ndarray): Array of shape (count, dimensions) containing the vectors to be quantized.
        num_bits (int): Number of bits for quantization.

    Returns:
        np.ndarray: Array of shape (count, dimensions) containing the quantized vectors.
    """
    # 1. Find the dynamic range of the data
    v_min = vectors.min()
    v_max = vectors.max()

    # 2. Calculate scale factor to fit range into 0-255
    scale = 255 / (v_max - v_min)

    # 3. Compress: Shift, Scale, Round, and Clip to 8-bit integers
    quantized_int8 = np.round((vectors - v_min) * scale)
    quantized_int8 = np.clip(quantized_int8, 0, 255).astype(np.uint8)

    # 4. Decompress: Convert back to Float (Lossy approximation)
    reconstructed = (quantized_int8.astype(np.float32) / scale) + v_min

    # 5. Renormalize (Crucial step in vector search engines)
    norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
    return reconstructed / norms


def run_experiment():
    print(f" STARTING QUANTIZATION EXPERIMENT")
    print(f"   Dimensions: {DIMENSIONS}")
    print(f"   Database Size: {NUM_VECTORS}")
    print("-" * 50)

    # 1. Generate Data
    print("1. Generating Synthetic Embeddings...")
    database = generate_vectors(NUM_VECTORS, DIMENSIONS)
    query = generate_vectors(1, DIMENSIONS)  # Single query vector

    # 2. Quantize Data
    print("2. Compressing Vectors (Float32 -> Int8)...")
    start_time = time.time()
    database_quant = simulate_quantization(database)
    query_quant = simulate_quantization(query)
    print(f"   Compression simulated in {time.time() - start_time:.4f}s")

    # 3. Ground Truth Search (Float32)
    print("3. Running EXACT Search (Float32)...")
    # Dot product of normalized vectors = Cosine Similarity
    true_scores = np.dot(database, query.T).flatten()
    # Get the indices of the top K results
    true_top_indices = np.argsort(true_scores)[::-1][:TOP_K]

    # 4. Approximate Search (Int8)
    print("4. Running APPROXIMATE Search (Int8)...")
    quant_scores = np.dot(database_quant, query_quant.T).flatten()
    quant_top_indices = np.argsort(quant_scores)[::-1][:TOP_K]

    # 5. Analysis
    print("-" * 50)
    print("RESULTS ANALYSIS")

    # Recall: How many of the True Top 10 did we find in the Approx Top 10?
    intersection = len(set(true_top_indices).intersection(set(quant_top_indices)))
    print(f"Intersection Rate: {intersection}")
    recall = (intersection / TOP_K) * 100

    # Error: How much did the score drift?
    score_diff = np.abs(true_scores - quant_scores)
    avg_error = np.mean(score_diff)
    max_error = np.max(score_diff)

    print(f"   Recall @ {TOP_K}: {recall:.0f}%  (Did we find the right documents?)")
    print(f"   Avg Score Error: {avg_error:.6f} (How much did similarity drift?)")
    print(f"   Max Score Error: {max_error:.6f} (Worst case drift)")
    print("-" * 50)

    if recall == 100:
        print("CONCLUSION: Int8 Quantization is SAFE. No accuracy loss detected.")
    elif recall > 90:
        print("CONCLUSION: Minor accuracy loss. Acceptable for high performance.")
    else:
        print("CONCLUSION: significant accuracy loss. Do not use quantization.")


if __name__ == "__main__":
    run_experiment()
