# syntax=docker/dockerfile:1
# Build:  docker build -t quantagent .
# Test:   docker build --target test .        (ruff + pytest + evaluation on the synthetic fixture)
# Use:    docker run --rm -v qa:/app/var quantagent setup        (download real data, ingest)
#         docker run --rm -v qa:/app/var quantagent tool rank_funds '{"metric": "sharpe", "period": "3y"}'
#         docker run -i --rm -v qa:/app/var quantagent serve      (MCP over stdio)
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data/reference ./data/reference

FROM base AS test
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs
RUN pip install -e ".[dev]" \
 && ruff check src scripts tests \
 && pytest -q --cov=quantagent --cov-fail-under=90 \
 && quantagent eval --fixture --out-dir /tmp/eval

FROM base AS runtime
RUN pip install ".[ml,agent]" && useradd --create-home --uid 10001 quantagent \
 && mkdir -p /app/var && chown quantagent /app/var
USER quantagent
# Everything the container writes lives under /app/var (mount a volume there).
ENV QUANTAGENT_DB_URL=sqlite:////app/var/db/quantagent.db \
    QUANTAGENT_STAGING_DIR=/app/var/staging \
    QUANTAGENT_EXTERNAL_DIR=/app/var/external \
    QUANTAGENT_MODEL_DIR=/app/var/models \
    QUANTAGENT_REPORTS_DIR=/app/var/reports
VOLUME ["/app/var"]
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 CMD ["quantagent-server", "--healthcheck"]
ENTRYPOINT ["quantagent"]
CMD ["serve"]
