import os
import uuid
from dotenv import load_dotenv
import json

load_dotenv(override=True)
# Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 60))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 0))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 0))  # Seconds
MAX_CONCURRENT_BATCHES = int(os.getenv("MAX_CONCURRENT_BATCHES", 1))
COMPLETION_WINDOW = "24h"

client_id = str(uuid.uuid4())

api_key_tier1 = os.getenv("OPENAI_API_KEY_TIER1")
api_key_tier5 = os.getenv("OPENAI_API_KEY_TIER5")
# OPENAI_KEYS = [api_key_tier1, api_key_tier5]

# I/O Path
INPUT_PATH = os.getenv("INPUT_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

OPENAI_KEYS = [api_key_tier5]

MODEL="openai/gpt-oss-120b"

API_KEYS_PATH = os.getenv("API_KEYS_PATH")
with open(API_KEYS_PATH, "r", encoding='utf-8') as fl:
    api_keys_mapping = json.load(fl)

NVIDIA_API_KEYS = []
for x in api_keys_mapping['api_keys'].values():
    NVIDIA_API_KEYS.extend(x)

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", "input_data")