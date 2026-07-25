# ---- builder -------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN pip install poetry \
 && poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock* ./
RUN poetry lock \
 && poetry install --only main --no-root --no-interaction

COPY ticketiq/ ./ticketiq/
COPY README.md ./
RUN poetry install --only main --no-interaction

# ---- runtime -------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    LLM_MODE=mock \
    DATA_DIR=/app/data

COPY --from=builder /build/.venv /app/.venv

WORKDIR /app
COPY ticketiq/ ./ticketiq/
COPY data/kb/ ./data/kb/
COPY data/tickets.jsonl ./data/
COPY scripts/ ./scripts/

# Train the classifier at build time so the first request is not slowed by it.
RUN python scripts/train_classifier.py --save --seeds 3 \
 && useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "ticketiq.main:app", "--host", "0.0.0.0", "--port", "8000"]
