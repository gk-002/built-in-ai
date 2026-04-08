# ── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="Ganesh Kendre <myslfgk24@gmail.com>"
LABEL description="Supply Chain Inventory Management - OpenEnv Submission"
LABEL version="1.0.0"

# ── Environment Variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime inference variables (override at container run time)
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
ENV HF_TOKEN=""

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Source ───────────────────────────────────────────────────────────────
COPY environment.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# ── Health Check ─────────────────────────────────────────────────────────────
# Validates that reset() returns 200 when pinged
COPY app.py .
EXPOSE 7860

# ── Default Command ───────────────────────────────────────────────────────────
# Starts the FastAPI server for HuggingFace Spaces
CMD ["python", "app.py"]
