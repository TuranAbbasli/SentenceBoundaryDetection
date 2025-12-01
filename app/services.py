import logging
from typing import Optional

import numpy as np
import tritonclient.grpc as grpcclient

from azcharboundary.utils.constants import TERMINAL_SENTENCE_CHAR_LIST, SENTENCE_TAG
from azcharboundary.utils.features import FeatureExtractor

from .config import TRITON_URL, MODEL_NAME, NETWORK_TIMEOUT, INPUT_NAME, OUTPUT_NAME, FEATURE_DIM

logger = logging.getLogger("AZSBDService")

# Global instances
feature_extractor = FeatureExtractor()
triton_client: Optional[grpcclient.InferenceServerClient] = None


def get_triton_client() -> Optional[grpcclient.InferenceServerClient]:
    """Get the global Triton client instance."""
    return triton_client


def set_triton_client(client: grpcclient.InferenceServerClient) -> None:
    """Set the global Triton client instance."""
    global triton_client
    triton_client = client


def warmup_model(client: grpcclient.InferenceServerClient, num_requests: int = 5) -> None:
    """Warm up the Triton model to avoid cold-start effects."""
    logger.info(f"Warming up model '{MODEL_NAME}' with {num_requests} dummy requests...")
    
    features = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    inp = grpcclient.InferInput(INPUT_NAME, [1, FEATURE_DIM], "FP32")
    inp.set_data_from_numpy(features)
    out = grpcclient.InferRequestedOutput(OUTPUT_NAME)

    for _ in range(num_requests):
        client.infer(
            model_name=MODEL_NAME,
            inputs=[inp],
            outputs=[out],
            client_timeout=NETWORK_TIMEOUT,
            headers={},
        )
    
    logger.info("Warmup complete.")


def preprocessing(
    text: str,
    left_window: int = 9,
    right_window: int = 9,
) -> tuple[Optional[list[grpcclient.InferInput]], Optional[list[grpcclient.InferRequestedOutput]], list[int]]:
    """Extract features and prepare Triton inference inputs."""
    terminal_indices: list[int] = [
        i for i, char in enumerate(text) if char in TERMINAL_SENTENCE_CHAR_LIST
    ]

    if not terminal_indices:
        return None, None, []

    terminal_features = feature_extractor.get_char_features(
        text,
        left_window,
        right_window,
        positions=terminal_indices,
    )

    features_np = np.asarray(terminal_features, dtype=np.float32)

    if features_np.ndim == 1:
        features_np = features_np.reshape(-1, FEATURE_DIM)

    if features_np.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {FEATURE_DIM}, got {features_np.shape[1]}"
        )

    n_rows, n_cols = features_np.shape

    inp = grpcclient.InferInput(INPUT_NAME, [n_rows, n_cols], "FP32")
    inp.set_data_from_numpy(features_np)

    out = grpcclient.InferRequestedOutput(OUTPUT_NAME)

    return [inp], [out], terminal_indices


def postprocessing(
    text: str,
    positions: list[int],
    response: Optional[grpcclient.InferResult],
) -> tuple[str, int]:
    """Insert sentence boundary tags based on model predictions."""
    if response is None or not positions:
        return text, 0

    probs = response.as_numpy(OUTPUT_NAME)
    if probs is None:
        raise RuntimeError(
            f"No output received from Triton for output name '{OUTPUT_NAME}'"
        )

    labels_np = (probs[:, 1] > probs[:, 0]).astype(np.int32)
    labels: list[int] = labels_np.tolist()

    result = list(text)

    tag_shift = 1
    boundary_count = 0
    for prediction, terminal_idx in zip(labels, positions):
        if prediction:
            result.insert(terminal_idx + tag_shift, SENTENCE_TAG)
            tag_shift += 1
            boundary_count += 1

    return "".join(result), boundary_count
