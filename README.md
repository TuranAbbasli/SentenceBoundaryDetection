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
           └─ checkpoint.zip   # unpacked to checkpoint.tl by the model-init service

`docker-compose.yml` (repo root) mounts `triton_inference_server/` as the model repository.

### Starting Everything

From the repo root:

    docker compose up -d --build

This runs three services in order:

1. `model-init` — unpacks `checkpoint.zip` into `checkpoint.tl` (one-off, idempotent;
   `*.tl` is gitignored so a fresh clone only has the zip)
2. `triton` — loads the model, gRPC on host port 8001
3. `api` — FastAPI, host port 8000, waits until Triton reports healthy

Then segment text:

    curl -X POST http://localhost:8000/segment \
      -H 'Content-Type: application/json' \
      -d '{"text": "Bu birinci cümlədir. Bu isə ikinci cümlədir."}'

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
