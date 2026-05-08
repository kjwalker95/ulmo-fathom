# Phase 0 demo containerization.


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY apis/ ./apis/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "scripts/sanity_check_lines.py"]
CMD ["--help"]
