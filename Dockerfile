FROM python:3.12-slim

# Install system build tools required to compile native C++ wheels inside amd64 Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv globally inside the image
RUN pip install uv --no-cache-dir

# Copy dependency records
COPY pyproject.toml uv.lock ./

# Compile wheels and sync dependencies with explicitly turned off hardware backends for standard CPU
RUN uv sync --frozen --no-install-project

# Copy your source code and local models directory
COPY . .

# Setup target evaluation directories
RUN mkdir -p /input /output /tmp/models

# Transfer model weights into the container tier
COPY models/qwen2.5-0.5b-instruct-q4_k_m.gguf /tmp/models/

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "main.py"]