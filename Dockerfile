FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir ".[service,models]"

RUN groupadd --system app && useradd --system --gid app --home /app app \
    && mkdir -p /app/runtime && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "productmatch.api.service:app", "--host", "0.0.0.0", "--port", "8000"]
