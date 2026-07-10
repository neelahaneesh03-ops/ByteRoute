# ByteRoute: Token-Efficient Routing Agent

ByteRoute is an asynchronous, hybrid routing agent built for Track 1 of the AMD Developer Hackathon (Act II). It minimizes Fireworks AI token consumption by processing deterministic tasks locally while optimizing cloud inference pipelines.

## Architecture

- **Layer 1 (Local Interception - 0 Tokens):** Captures baseline mathematical equations and clear, single-vector sentiment queries inside the local container environment using optimized regular expressions and keyword heuristics. These resolve at exactly zero token cost to the leaderboard score.
- **Layer 2 (Local LLM - 0 Tokens):** Named entity recognition runs entirely inside the container using a bundled Qwen2.5-0.5B-Instruct GGUF model via llama.cpp. This handles a full linguistic task with zero API calls, keeping the token score untouched.
- **Layer 3 (Remote):** Tricky or complex linguistic tasks (code generation, code debugging, summarization, mathematical word problems, logical reasoning, factual knowledge, and mixed sentiment) fall back to the Fireworks AI infrastructure. To combat conversational verbosity, ByteRoute injects category-specific system instructions (e.g., clamping summarization to 25 words or forcing raw markdown code blocks for debugging) to severely truncate output token usage.
- **Sequential Execution:** Because local model inference is CPU-bound, tasks are processed sequentially through a shared async HTTP client, which keeps the whole run well within the 10-minute runtime limit while avoiding memory contention on the 4GB/2vCPU grading environment.

## Run via Docker (Recommended)

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY="your_api_key" \
  -e FIREWORKS_BASE_URL="your_base_url" \
  -e ALLOWED_MODELS="your_model_list" \
  haneeshneela/amd_routing_agent:latest
```

## Local Development

1. Install dependencies: `uv sync`
2. Configure environment variables in a `.env` file
3. Run: `uv run python main.py`

## Input/Output

- Input: `/input/tasks.json`
- Output: `/output/results.json`

## Tech Stack

Python 3.12, asyncio, httpx, Docker (linux/amd64), Fireworks AI, GLM 5.2, uv
