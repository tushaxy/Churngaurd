# ==============================================================================
# ChurnGuard Streamlit Frontend Dockerfile
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Stage
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY config.yaml .
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8501

ENV FASTAPI_URL=http://api:8000
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=true

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "app/Home.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
