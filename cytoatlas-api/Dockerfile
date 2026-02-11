# Stage 1: Build React frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend dependency files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --ignore-scripts

# Copy frontend source
COPY frontend/ .

# Build frontend (outputs to ../static/)
RUN npm run build

# Stage 2: Build Python backend
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv pip install -e .

# Stage 3: Production
FROM python:3.11-slim AS production

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Copy built frontend assets (overwrite static/)
COPY --from=frontend-builder --chown=appuser:appuser /app/static/ /app/static/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
