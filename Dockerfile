# Voyage Analytics 2.0 — FastAPI Dockerfile
# Multi-stage build for minimal production image

#  Stage 1: Build 
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


#  Stage 2: Runtime 
FROM python:3.12-slim AS runtime

LABEL maintainer="Voyage Analytics Team"
LABEL version="2.0.0"
LABEL description="Voyage Analytics 2.0 — Travel Intelligence API"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY .env.example .env

# Create logs directory
RUN mkdir -p logs

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
