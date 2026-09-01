import os

# Configuration
# TRITON_URL defaults to localhost for running the API directly on the host;
# docker compose overrides it with the triton service name.
TRITON_URL: str = os.getenv("TRITON_URL", "localhost:8001")
MODEL_NAME: str = os.getenv("MODEL_NAME", "model")
NETWORK_TIMEOUT = 60.0
INPUT_NAME = "input__0"
OUTPUT_NAME = "output__0"
FEATURE_DIM = 28
