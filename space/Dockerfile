# Slothing web zhuyin IME demo. CPU-only: llama-cpp-python runs the 34M
# SlothLM GGUF (downloaded from the Hub at startup); FastAPI serves the IME UI
# and a /api/decode endpoint that mirrors slothingd's grammar-constrained decode.
FROM python:3.11-slim

# llama-cpp-python needs a C/C++ toolchain to build its CPU wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces run as a non-root user; keep the model cache writable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py .
COPY --chown=user phonetic_table.tsv .
COPY --chown=user static ./static

USER user
EXPOSE 7860
# Diagnostic boot: print markers so we can see exactly where startup dies
# (import hang vs uvicorn hang) in the Space run logs.
CMD ["sh", "-c", "echo BOOT_MARKER; python -c 'import sys; print(\"PY\", sys.version, flush=True)'; python -c 'import app; print(\"APP_IMPORT_OK\", flush=True)'; echo LAUNCHING_UVICORN; exec uvicorn app:app --host 0.0.0.0 --port 7860 --log-level info"]
