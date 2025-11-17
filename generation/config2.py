"""
Global configuration for the async data generation framework.

Reads most settings from environment variables with safe defaults.
Prefer environment variables for secrets and deployment customization.
"""
import os
import json
import json_schemas

llm_response_schema = json_schemas.SBD_SCHEMA

# ---- API / Model ----
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

API_KEYS_PATH = os.getenv("API_KEYS_PATH")
with open(API_KEYS_PATH , "r", encoding='utf-8') as fl:
    api_keys_mapping = json.load(fl)

API_KEYS  = []
for x in api_keys_mapping['api_keys'].values():
    API_KEYS.extend(x)
            
MODEL_ID = os.getenv("NVIDIA_MODEL_ID", "openai/gpt-oss-120b")

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_P = float(os.getenv("TOP_P", "1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "40000"))


# ---- Concurrency ----
# Default workers = 20 per configured API key (minimum 20 if no keys yet).
DEFAULT_WORKERS = 10 * max(1, len(API_KEYS) or 1)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(DEFAULT_WORKERS)))

# ---- SQLite ----
DB_PATH = os.getenv("DB_PATH", "completions.sqlite")

# ---- Input/Output ----
INPUT_PATH = os.getenv("INPUT_PATH", "input.jsonl")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "900"))  # batch size for DB existence checks

# ---- Retries / Backoff ----
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", "4.0"))

if __name__ == "__main__":
    print(f'API_keys count: {len(API_KEYS)}\n\n')
    print(f'API_keys: {API_KEYS}')