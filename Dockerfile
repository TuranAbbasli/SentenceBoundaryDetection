FROM python:3.11-slim

# uv for installs (project convention; also much faster than pip here)
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /srv

COPY requirements-api.txt .
RUN uv pip install --system -r requirements-api.txt

COPY app ./app
COPY azcharboundary ./azcharboundary

RUN useradd --create-home --uid 10001 sbd
USER sbd

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
