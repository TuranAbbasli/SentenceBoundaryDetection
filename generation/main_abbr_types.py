# main2.py
"""
Main batch-based script for ABBREVIATION + NON-TERMINAL-PERIOD classification
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Optional
import aiofiles
from model import BatchAbbrvOutput


from config import (
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_CONCURRENT_BATCHES,
    INPUT_PATH,
    OUTPUT_PATH,
)
from nvidia_client import NvidiaClient
from logging_setup import logger
from prompts.prompt_abbrv import ASSISTANT_TEMPLATE, PROMPT_TEMPLATE

STOP = object()
MAX_RETRIES_NVIDIA_CLIENT = 10

batch_producer_logger = logging.getLogger("app_logger.batch_producer")
batch_consumer_logger = logging.getLogger("app_logger.batch_consumer")


# -------------------------------
# Batch helper
# -------------------------------
def get_batches(data: List[str], batch_size: int):
    """
    Convert list of text chunks into list:
        [(0, [text1, text2, ...]), (1, [...])]
    """
    return [
        (batch_num, data[i:i + batch_size])
        for batch_num, i in enumerate(range(0, len(data), batch_size))
    ]


# -------------------------------
# Parse model output
# -------------------------------

def process_response(resp, consumer_id: int) -> Optional[List[Dict]]:
    log = logging.getLogger(f"app_logger.model_call.{consumer_id}")

    content = (
        resp.choices[0].message.content
        if (resp.choices and resp.choices[0].message and resp.choices[0].message.content)
        else ""
    )

    if not content:
        log.warning("Empty content from API.")
        return None

    try:
        raw = json.loads(content)

        validated = BatchAbbrvOutput(raw)

        # validated.__root__ is a list[AbbrvOutput]
        return [item.model_dump() for item in validated.root]

    except Exception as e:
        log.error(f"Schema validation failed: {e}")
        return None



# -------------------------------
# Call model
# -------------------------------
async def call_model(nv: NvidiaClient, batch: List[str], consumer_id: int):
    log = logging.getLogger(f"app_logger.model_call.{consumer_id}")

    # Prepare prompt
    user_prompt = PROMPT_TEMPLATE.replace("{chunk}", json.dumps(batch, ensure_ascii=False))

    messages = [
        {"role": "system", "content": ASSISTANT_TEMPLATE},
        {"role": "user", "content": user_prompt},
    ]

    resp = await nv.chat_completion(messages=messages)

    decoded = process_response(resp, consumer_id)

    if isinstance(decoded, list):
        log.info("Batch processed.")
        return decoded

    return None


# -------------------------------
# Producer
# -------------------------------
async def batch_producer(queue: asyncio.Queue, batches):
    batch_producer_logger.info("Batch producer started.")

    for batch in batches:
        await queue.put(batch)

    batch_producer_logger.info("Producer finished enqueuing.")


# -------------------------------
# Consumer
# -------------------------------
async def batch_consumer(consumer_id, queue, nv, out_path):
    log = logging.getLogger(f"app_logger.batch_consumer.{consumer_id}")
    log.info("Consumer started.")

    try:
        while True:
            item = await queue.get()

            if item is STOP:
                queue.task_done()
                break

            batch_index, batch_texts = item

            # Model call
            output = None
            for _ in range(MAX_RETRIES_NVIDIA_CLIENT):
                output = await call_model(nv, batch_texts, consumer_id)
                if output is not None:
                    break

            # Write results
            if output is not None:
                # One output per batch element
                for i, result in enumerate(output):
                    chunk_id = f"chunk_{batch_index * BATCH_SIZE + i + 1}"

                    to_write = {
                        "chunk": chunk_id,
                        "input": batch_texts[i],
                        "abbr": result.get("abbr", []),
                        "types": result.get("types", [])
                    }

                    await append_jsonl(out_path, to_write)

                log.info(f"Batch {batch_index} saved.")

            else:
                log.error(f"Batch {batch_index} failed.")
                await append_jsonl(out_path, {"batch": batch_index, "error": True})

            queue.task_done()

    finally:
        log.info("Consumer exited.")


# -------------------------------
# JSONL writer
# -------------------------------
async def append_jsonl(path: str, item: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(json.dumps(item, ensure_ascii=False) + "\n")


# -------------------------------
# Main
# -------------------------------
async def main():
    logger.info("App started.")

    # Load input
    texts = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                texts.append(obj["text"])
            except Exception as e:
                logger.error(f"Invalid JSON on line {idx}: {e}")

    batches = get_batches(texts, BATCH_SIZE)
    logger.info(f"Prepared {len(batches)} batches.")

    # Queues + Client
    queue = asyncio.Queue(maxsize=MAX_CONCURRENT_BATCHES)
    nv = NvidiaClient()

    # Create tasks
    producer = asyncio.create_task(batch_producer(queue, batches))
    consumers = [
        asyncio.create_task(batch_consumer(i + 1, queue, nv, OUTPUT_PATH))
        for i in range(MAX_CONCURRENT_BATCHES)
    ]

    # Run
    await producer

    for _ in consumers:
        await queue.put(STOP)

    await queue.join()
    await asyncio.gather(*consumers)

    await nv.close()

    logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
