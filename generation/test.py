import asyncio
import platform
import os
import time
from collections import defaultdict
from openai import AsyncOpenAI
from openai import OpenAIError
import json

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 1. Make sure you have your NVIDIA key and endpoint set
#    export OPENAI_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxxxxxx"
#    export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"

with open("generation/api_keys.json" , "r", encoding='utf-8') as fl:
    api_keys_mapping = json.load(fl)

NVIDIA_API_KEYS = []
for x in api_keys_mapping['api_keys'].values():
    NVIDIA_API_KEYS.extend(x)

messages = [
    {"role": "system", "content": "You are a test assistant."},
    {"role": "user", "content": "Say 'Hello from NVIDIA NIM!' and give me a random number."}
]

async def test_api_keys():
    sdk_bug = defaultdict(int)
    fail = defaultdict(int)

    for i in range(1):
        print(f'\n\nLoop count: {i+1}\n\n')
        for key in NVIDIA_API_KEYS:
            client = AsyncOpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1")
            try:
                resp = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    max_tokens=500,
                )

                print("✅ Request succeeded.")

                if resp is None:
                    print("Error: resp is None.")
                    continue

                if not getattr(resp, "choices", None):
                    print("Error: resp.choices is None or empty—possible provider error or SDK bug.\n")
                    print(f'SDK bug: {key}')

                    sdk_bug[key] += 1

                    # Dumping raw resp might offer clues
                    try:
                        print("Raw response:", resp.model_dump_json())
                        print("\n\n")
                    except Exception:
                        pass
                    continue

                # Safe to access
                content = resp.choices[0].message.content
                print("Response:", content)

                if resp.usage:
                    print("Prompt tokens:", resp.usage.prompt_tokens)
                    print("Completion tokens:", resp.usage.completion_tokens)
                    print("Total tokens:", resp.usage.total_tokens)
                    print("\n\n")

            except OpenAIError as e:
                fail[key] += 1
                print(f"❌ API request failed: {str(e)}.\nAPI key: {key}\n\n")
            except Exception as e:
                fail[key] += 1
                print(f"❌ Unexpected error: {str(e)}.\nAPI key: {key}\n\n")

    print(f'\n\nSDK bug: {sdk_bug}\n\n')
    print(f'\n\nFail: {fail}\n\n')

asyncio.run(test_api_keys())