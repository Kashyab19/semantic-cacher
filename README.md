# Semantic Cacher

A simple semantic caching layer for LLM responses using vector embeddings and Qdrant. Cache semantically similar queries to reduce API costs and latency.


## Architecture

```
User Query → Embedding Model → Vector Search → Cache Hit/Miss
                                            ↓
                                    LLM API (Ollama)
                                            ↓
                                    Store in Cache
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Ollama running on `localhost:11434` with `qwen2.5:0.5b` model

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd semantic-cacher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant**
   ```bash
   docker-compose up -d
   ```

4. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ollama pull qwen2.5:0.5b
   ```

5. **Run the API server**
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## Usage

### API Endpoints

**POST `/chat`**
- Request body: `{"prompt": "Your question here"}`
- Response: `{"response": "...", "source": "cached|llm", "latency": 0.123}`

**GET `/health`**
- Check if the service is running and models are loaded

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

### Benchmark

Run the benchmark script to test cache performance:

```bash
python scripts/benchmark.py
```

This will send a series of queries (some repeated) and show cache hit rates and latency improvements.


## How It Works

1. **Query embedding**: Incoming queries are converted to 384-dimensional vectors using `all-MiniLM-L6-v2`
2. **Similarity search**: Qdrant searches for similar vectors in the cache
3. **Cache decision**: If similarity > 0.85, return cached response; otherwise call LLM
4. **Cache storage**: New LLM responses are stored with their query embeddings

## Tech Stack

- **FastAPI** - Web framework
- **Qdrant** - Vector database
- **Sentence Transformers** - Embedding generation
- **Ollama** - Local LLM inference

## License

MIT
