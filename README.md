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

## 🔮 Future Roadmap

Now that ByteRoute has evolved beyond its hackathon constraints, the following architectural upgrades are planned to transform it from a functional prototype into a robust, general-purpose LLM Routing Middleware:

### 1. Shift from Heuristic to Learned Routing (Embedding Classifier)
* **Objective:** Replace static regex/keyword rules with a semantic intent classifier to remove manual maintenance overhead.
* **Scope:** Implement a micro-embedding model (e.g., a tiny, 10ms local sentence encoder) to map incoming prompt vectors against semantic clusters (such as `DevOps`, `Data Extraction`, or `Code Optimization`). Prompts will be dynamically dispatched to the optimal tier based on real-time complexity scoring rather than brittle string matching.

### 2. Implement Outcome-Based Cascades (Validation Loops)
* **Objective:** Stop guessing whether a local model can handle a nuanced task—let it try, and programmatically verify the output.
* **Scope:** Build a localized Context-Action-Feedback (C-A-F) loop. For highly structured tasks like JSON schema mapping, the local Qwen-0.5B instance will execute the initial action. A rule-based local verifier will inspect the response structure; if a syntax or schema validation exception triggers, the system will transparently escalate the task to the cloud tier.

### 3. Input Trimming & Context Optimization Engine
* **Objective:** Strip out unnecessary token bloat before payloads ever hit the network layer.
* **Scope:** Build a pre-dispatch optimization middleware. The engine will programmatically minify tool schemas, compress repetitive chat history, strip redundant whitespaces, and pack sparse JSON arrays into dense columnar structures, reducing outbound cloud token consumption by an additional 30% to 70% with zero semantic loss.

### 4. Continuous Performance Ledger (Stateful Session Memory)
* **Objective:** Enable the router to learn organically from execution-grounded experience.
* **Scope:** Integrate a low-latency local storage backend (SQLite/Redis) to act as a permanent performance ledger. By logging transaction metrics like token throughput, latency variations, and verification pass/fail flags, the orchestrator can dynamically recalibrate its routing confidence parameters based on running historical data.

### 5. Multi-Model Consensus & Variant Fusion
* **Objective:** Match frontier-class accuracy thresholds using an aggregated budget panel of smaller open-weight models.
* **Scope:** Implement an algorithmic ensemble layer. For highly sensitive reasoning or code review pipelines, the router will query multiple allowed lightweight models concurrently and pass the results to a localized consensus judge to synthesize a single optimized response at a fraction of the cost of a premier frontier model.

### 6. Edge Infrastructure & Low-Resource Stress Testing
* **Objective:** Validate real-world deployability on genuinely constrained environments, moving past "does it technically run on a high-spec development machine."
* **Scope:** Profile peak Resident Set Size (RSS) memory utilization of the local GGUF layer under heavy concurrent load rather than resting footprint. Execute automated benchmarks against severely throttled environments (`docker run --memory=2g --cpus=1`) and physical edge nodes (e.g., Raspberry Pi) to pinpoint the precise crossover threshold where local validation cycles cost more in latency than a direct API call.