# Production Multi-Stage Dockerfile for Seattle House Price Prediction API
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies for LightGBM/C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Install runtime OpenMP library required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Create non-root security user
RUN useradd -m -u 1001 appuser

# Copy application code, model artifacts, and reference data
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/raw/zipcode_demographics.csv /app/data/raw/zipcode_demographics.csv

# Permissions
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
