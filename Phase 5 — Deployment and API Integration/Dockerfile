# syntax=docker/dockerfile:1

# ---- Builder: install dependencies into a venv ----
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Runtime: minimal image with the venv + app ----
FROM python:3.13-slim AS runtime

# Non-root user for least privilege.
RUN useradd --create-home --uid 10001 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/app/models \
    PREDICTION_LOG_PATH=/var/log/hospital-ml/predictions.log

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/
COPY models/ ./models/

# Writable log directory owned by the non-root user.
RUN mkdir -p /var/log/hospital-ml && chown -R appuser:appuser /var/log/hospital-ml /app

USER appuser
EXPOSE 8000

# Container-level health check (ECS also probes /health via the target group).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
