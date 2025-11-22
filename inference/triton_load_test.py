# triton_merge_test_fixed.py

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tritonclient.grpc as grpcclient

from azcharboundary.utils.constants import TERMINAL_SENTENCE_CHAR_LIST
from azcharboundary.utils.features import FeatureExtractor


# CONFIG
TRITON_URL = "localhost:8001"
MODEL_NAME = "model"

INPUT_NAME = "input"
OUTPUT_NAME = "output"
FEATURE_DIM = 19
NETWORK_TIMEOUT = 60.0

CONCURRENCY = 10                                   # fixed concurrency
  
BATCH_SIZE = 1                                     # how many chunks merged into ONE Triton request
BASE_TOTAL_CHUNKS = 100_000                        # target total logical chunks
NUM_REQUESTS = BASE_TOTAL_CHUNKS // BATCH_SIZE     # Triton requests to send

WARMUP_REQUESTS = 10

feature_extractor = FeatureExtractor()


# PREPROCESSING
def extract_features(text: str) -> np.ndarray:
    """Return array of shape (rows_per_chunk, FEATURE_DIM)."""
    indices = [i for i, ch in enumerate(text) if ch in TERMINAL_SENTENCE_CHAR_LIST]

    if not indices:
        return np.empty((0, FEATURE_DIM), np.float32)

    feats = feature_extractor.get_char_features(text, 5, 5, positions=indices)
    arr = np.asarray(feats, np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, FEATURE_DIM)

    return arr

def build_io(feature_block: np.ndarray):
    inp = grpcclient.InferInput(INPUT_NAME, list(feature_block.shape), "FP32")
    inp.set_data_from_numpy(feature_block)
    out = grpcclient.InferRequestedOutput(OUTPUT_NAME)
    return [inp], [out]

# WARMUP
def warmup(client, merged_block: np.ndarray):
    print(f"\nWarming up with {WARMUP_REQUESTS} requests...")
    inputs, outputs = build_io(merged_block)
    for _ in range(WARMUP_REQUESTS):
        client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            client_timeout=NETWORK_TIMEOUT,
        )
    print("Warmup done.\n")

# BATCH Processing
def batch_processing(client, text: str):
    print("\n=======================================")
    print(f" BATCH Processing — BATCH_SIZE = {BATCH_SIZE}")
    print("=======================================\n")

    # 1) Extract features for one logical text chunk
    t0 = time.time()
    base_features = extract_features(text)
    t1 = time.time()
    pre_ms = (t1 - t0) * 1000

    rows_per_chunk = base_features.shape[0]

    if rows_per_chunk == 0:
        print("No terminals found — nothing to test.")
        return

    # 2) Compute shapes and totals
    total_logical_chunks = NUM_REQUESTS * BATCH_SIZE
    batch = np.tile(base_features, (BATCH_SIZE, 1))  # (rows_per_chunk * BATCH_SIZE, D)
    rows_per_request = batch.shape[0]
    total_rows = total_logical_chunks * rows_per_chunk

    print(f"Rows per chunk:            {rows_per_chunk}")
    print(f"BATCH_SIZE (chunks/req):   {BATCH_SIZE}")
    print(f"NUM_REQUESTS (Triton):     {NUM_REQUESTS}")
    print(f"Total logical chunks:      {total_logical_chunks}")
    print(f"Rows per Triton request:   {rows_per_request}")
    print(f"Total rows processed:      {total_rows}")
    print(f"Preprocessing time (1x):   {pre_ms:.2f} ms")

    # 3) Warmup with one merged request shape
    warmup(client, batch)

    # 4) Worker sending one merged request
    def worker(_):
        inputs, outputs = build_io(batch)
        s = time.time()
        client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            client_timeout=NETWORK_TIMEOUT,
        )
        e = time.time()
        return (e - s) * 1000.0  # ms

    # 5) Run load test
    latencies = []
    wall0 = time.time()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(worker, i) for i in range(NUM_REQUESTS)]
        for f in as_completed(futures):
            latencies.append(f.result())

    wall1 = time.time()
    wall_ms = (wall1 - wall0) * 1000.0

    # 6) Stats
    latencies.sort()
    avg_ms = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    rows_per_sec = total_rows / (wall_ms)

    print("\n--- RESULTS ---")
    print(f"Wall time:              {wall_ms:.2f} ms")
    print(f"Rows/ms:                {rows_per_sec:.2f}")
    print(f"Avg Triton req latency: {avg_ms:.2f} ms")
    print(f"P50 latency:            {p50:.2f} ms")
    print(f"P95 latency:            {p95:.2f} ms")
    print(f"P99 latency:            {p99:.2f} ms")
    print("=======================================\n")

# MAIN
def main():
    client = grpcclient.InferenceServerClient(
        url=TRITON_URL,
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
    )

    # optional old-client workaround:
    if not hasattr(client, "_stream"):
        client._stream = None  # type: ignore[attr-defined]

    try:
        if not client.is_server_live():
            raise RuntimeError("Triton server is not live")
        if not client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model '{MODEL_NAME}' is not ready")

        print("Connected to Triton via gRPC. Model is ready.")
        print(f"Using endpoint: {TRITON_URL}")

        base_text = (
            "Azərbaycan iqtisadiyyatı son illərdə sürətli inkişaf edir. "
            "Bu inkişaf müxtəlif sahələrdə özünü göstərir. "
            "Xüsusilə texnologiya, təhsil və enerji sektorunda ciddi dəyişikliklər var. "
            "Bir çox startaplar yaranır, dövlət innovasiyalara investisiya edir. "
            "Bu proses ölkənin rəqəmsal transformasiyasını daha da gücləndirir. "
            "Məhkəmə qərarı olmadan şəxsin telefon danışıqlarına nəzarət edilməsi qadağandır. "
            "Bu, vətəndaşların hüquq və azadlıqlarının qorunması üçün vacibdir. "
        )

        batch_processing(client, base_text)

    finally:
        client.close()


if __name__ == "__main__":
    main()
