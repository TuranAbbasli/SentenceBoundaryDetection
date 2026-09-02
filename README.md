# Sentence Boundary Detection

This project implements a custom Sentence Boundary Detection system using the structure of the CharBoundary library.

## Training

Training is handled through train.py. You can adjust hyperparameters directly inside the training call:

    training_metrics = segmenter.train(
        data=train_set,
        model_params={"n_estimators": 128, "max_depth": 32},
        sample_rate=0.001,
        left_window=9,
        right_window=9,
        threshold=0.75,
        use_feature_selection=False,
        max_features=20,
    )

Modify these parameters to train a new model. Window sizes, thresholds, sample rate, and model configuration all impact performance.

## Inference with NVIDIA Triton

Inference is served by NVIDIA Triton Inference Server, with a FastAPI service in front of it.
Both run under Docker Compose. Triton image version: 25.11

### Folder Structure

The Triton model repository lives in `triton_inference_server/`:

    triton_inference_server/
    └─ model/
       ├─ config.pbtxt
       └─ 1
           ├─ checkpoint.zip   # committed
           └─ checkpoint.tl    # unzip this yourself, once per host

`docker-compose.yml` (repo root) mounts `triton_inference_server/` as the model repository.

### One-off: unpack the checkpoint

`config.pbtxt` loads `checkpoint.tl`, but `*.tl` is gitignored — it is 172MB, over
GitHub's 100MB file limit — so a fresh clone carries only the zip. **Do this once per
host, before the first `docker compose up`:**

    cd triton_inference_server/model/1
    unzip checkpoint.zip
    cd -

Skip it and Triton starts but fails to load the model, with the reason buried in
`docker compose logs triton`.

### Starting Everything

From the repo root:

    docker compose up -d --build

This runs two services:

1. `triton` — loads the model, gRPC on host port 8001
2. `api` — FastAPI, host port 8000, waits until Triton reports healthy

Then segment text:

    curl -X POST http://localhost:8000/segment \
      -H 'Content-Type: application/json' \
      -d '{"text": "Bu birinci cümlədir. Bu isə ikinci cümlədir."}'

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /segment` | Segment text into sentences |
| `GET /health` | Liveness — 200 whenever the process answers; ignores Triton |
| `GET /ready` | Readiness — 200 only if Triton is reachable and the model is loaded, else 503 |

`docker compose ps` reports `api` as `healthy` based on `/ready`, so a Triton
outage shows up as `unhealthy` without the container being restarted.

Logs and teardown:

    docker compose logs -f api
    docker compose down

### Running the API outside Docker

`TRITON_URL` defaults to `localhost:8001`, so with only Triton in Docker:

    docker compose up -d triton
    uv venv --python 3.11 && uv pip install -r requirements.txt
    uv run uvicorn app.api:app --host 0.0.0.0 --port 8000

### GPU

The compose file runs Triton on CPU by default. On a GPU host, uncomment the `deploy:`
block under the `triton` service (requires the NVIDIA Container Toolkit).
