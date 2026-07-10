import os
import sys
import json
from pathlib import Path
import httpx
from dotenv import load_dotenv
import re
import asyncio
from llama_cpp import Llama

load_dotenv()

try:
    FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
    FIREWORKS_BASE_URL = os.environ["FIREWORKS_BASE_URL"]
    ALLOWED_MODELS = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
except KeyError as e:
    print(f"Critical Error: Missing environment variable: {e}")
    sys.exit(1)

INPUT_FILE = Path("/input/tasks.json")
OUTPUT_FILE = Path("/output/results.json")
MODEL_PATH = Path("/tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")

print("Loading local model...")
llm = Llama(model_path=str(MODEL_PATH), n_ctx=1024, verbose=False)
print("Local model ready.")

POSITIVE_KEYWORDS = {
    "love", "loved", "great", "excellent", "amazing", "fantastic", "wonderful",
    "awesome", "good", "best", "happy", "enjoy", "enjoyed", "perfect", "brilliant",
    "superb", "outstanding", "pleasant", "delightful", "incredible"
}
NEGATIVE_KEYWORDS = {
    "hate", "hated", "terrible", "awful", "horrible", "bad", "worst", "poor",
    "disappointing", "disappointed", "boring", "dull", "useless", "broken",
    "frustrating", "frustrated", "angry", "sad", "disgusting", "unacceptable"
}

# --- DEFINITIONS FIXED BEFORE INVOCATION ---

def run_local_llm(system_prompt: str, user_prompt: str) -> str:
    prompt_format = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    output = llm(prompt_format, max_tokens=150, stop=["<|im_end|>"], temperature=0.0)
    return output["choices"][0]["text"].strip()

def route_task_locally(prompt: str) -> str | None:
    prompt_lower = prompt.lower().strip()

    # Math regex interception
    is_math = any(w in prompt_lower for w in ["calculate", "solve", "plus", "minus", "divided by", "%"]) \
              or any(op in prompt for op in ["+", "-", "*", "/"])
    if is_math and len(prompt.split()) <= 15:
        math_match = re.search(r"(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)", prompt_lower)
        if math_match:
            n1, op, n2 = float(math_match.group(1)), math_match.group(2), float(math_match.group(3))
            if op == "+": r = n1 + n2
            elif op == "-": r = n1 - n2
            elif op == "*": r = n1 * n2
            elif op == "/" and n2 != 0: r = n1 / n2
            else: r = None
            if r is not None:
                clean_res = int(r) if isinstance(r, float) and r.is_integer() else round(r, 4)
                return f"The result of the calculation is {clean_res}."
        
        pct = re.search(r"(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)", prompt_lower)
        if pct:
            p, t = float(pct.group(1)), float(pct.group(2))
            r = (p / 100) * t
            clean_res = int(r) if r == int(r) else round(r, 4)
            return f"Calculation evaluates to {clean_res} because {p}% of {t} is {clean_res}."

    # Sentiment regex interception
    is_sentiment = any(w in prompt_lower for w in ["positive", "negative", "sentiment", "classify", "opinion"])
    if is_sentiment:
        if any(c in prompt_lower for c in ["but", "however", "yet", "although", "though"]):
            return None
        parts = re.split(r"[?:]", prompt, maxsplit=1)
        text = parts[-1].lower().strip() if len(parts) > 1 else prompt_lower
        words = set(re.findall(r"\b\w+\b", text))
        pos = words & POSITIVE_KEYWORDS
        neg = words & NEGATIVE_KEYWORDS
        if pos and not neg:
            detected = ", ".join(list(pos)[:3])
            return f"Positive. The text expresses a favorable opinion using words like: {detected}."
        if neg and not pos:
            detected = ", ".join(list(neg)[:3])
            return f"Negative. The text expresses an unfavorable opinion using words like: {detected}."

    return None

async def call_fireworks_api(client: httpx.AsyncClient, prompt: str, model_id: str, system_override: str = None) -> str:
    system_prompt = system_override or "Answer directly in 1-2 sentences max. No filler text."
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    response = await client.post(
        f"{FIREWORKS_BASE_URL.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"},
        timeout=45.0
    )
    if response.status_code != 200:
        raise Exception(f"API Error ({response.status_code}): {response.text}")
    return response.json()["choices"][0]["message"]["content"]

async def process_single_task(client: httpx.AsyncClient, task: dict) -> str:
    prompt = task.get("prompt", "")
    task_id = task.get("task_id", "")
    category = task.get("category", "unknown").lower()
    prompt_lower = prompt.lower()

    # Layer 1: Regex Interception (0 tokens)
    local = route_task_locally(prompt)
    if local is not None:
        return local

    # Layer 2: Safe Local Processing (0 tokens) -> NER ONLY with robust full-name examples
    if "entity" in category or "ner" in category or "extract" in prompt_lower:
        system_override = (
            "Extract all named entities. Print ONLY in this format: Entity (Type). One per line. Do not include any other text.\n"
            "CRITICAL Rule: Full names of people must remain as a single entity. Do not split first and last names.\n"
            "Example Input: 'Maria Sanchez joined Fireworks AI in Berlin last March.'\n"
            "Example Output:\n"
            "Maria Sanchez (Person)\n"
            "Fireworks AI (Organization)\n"
            "Berlin (Location)\n"
            "March (Date)"
        )
        return await asyncio.to_thread(run_local_llm, system_override, prompt)

    # Layer 3: High-Accuracy Cloud Escalation -> Summarization, Math, Logic, Sentiment, and Code
    else:
        target_model = ALLOWED_MODELS[0]
        
        # Code Debugging
        if "debug" in category or any(w in prompt_lower for w in ["bug", "fix the", "find and fix"]):
            override = "You are an expert software engineering assistant. Fix the bug in the code snippet. Provide the corrected implementation inside markdown code fences, followed by exactly one brief sentence explaining the fix. Keep it under 15 words total."
        
        # Code Generation
        elif "generation" in category or any(w in prompt_lower for w in ["write a python", "write a function", "implement", "code a"]):
            override = "You are an expert software engineer. Write the requested code or function block. Respond ONLY with the working functional Python code inside standard markdown code fences (```python ... ```). Do not include any usage examples, textual descriptions, or introductory explanations."
        
        # Summarization (Escalated to cloud to protect key detail retention)
        elif "summar" in category or any(w in prompt_lower for w in ["summarize", "summarise", "condense", "gist"]):
            override = "You are an ultra-concise summarization agent. Condense the text. Respond with exactly one direct sentence of maximum 25 words that retains all core factual details. No introductory filler words or preambles."

        # Sentiment Analysis (Forced to include a brief justification sentence)
        elif "sentiment" in category or any(w in prompt_lower for w in ["sentiment", "classify", "opinion"]):
            override = "Classify the sentiment of the text. State the label first (Positive, Negative, or Mixed) followed immediately by exactly one concise sentence justifying the classification."

        # Math Fallback (Complex problems that regex missed)
        elif "math" in category or "calculat" in prompt_lower or "solve" in prompt_lower:
            override = "You are a precise mathematical solver. Calculate the answer carefully step-by-step internally, but provide ONLY the final clear numerical answer along with a single direct sentence showing the calculation. Keep it extremely brief."
            
        # Logic / Puzzles
        elif "reason" in category or "logic" in category or "puzzle" in prompt_lower:
            override = "You are a precise classification and text processing agent. Provide your answer in 1-2 concise sentences max. If classifying, state the label first followed by a brief 1-sentence justification. Omit any preambles, introductions, or structural padding."
            
        # Default text/factual
        else:
            override = "Answer directly in 1-2 sentences max. No filler text."

        try:
            return await call_fireworks_api(client, prompt, target_model, override)
        except Exception as e:
            print(f"API error for {task_id}: {e}")
            return "Error processing task."

async def main():
    print("ByteRoute Hybrid Engine initiated...")

    if not INPUT_FILE.exists():
        print(f"Input file not found at {INPUT_FILE}.")
        sys.exit(1)

    with open(INPUT_FILE, "r") as f:
        tasks = json.load(f)

    print(f"Processing {len(tasks)} tasks...")

    async with httpx.AsyncClient() as client:
        results = []
        for task in tasks:
            task_id = task.get("task_id")
            print(f"Processing: {task_id}")
            answer = await process_single_task(client, task)
            results.append({"task_id": task_id, "answer": answer})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("Processing completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())