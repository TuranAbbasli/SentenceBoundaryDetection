# main.py
"""
Main training script for abbreviation detection
"""
import asyncio
import json
import logging
import os
import random
from typing import Dict, List, Optional
import aiofiles
from pydantic import ValidationError

from config import (
    BATCH_SIZE,
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_CONCURRENT_BATCHES,
    INPUT_PATH,
    OUTPUT_PATH
)
from model import Spelling_Correction_Output
from nvidia_client import NvidiaClient
from logging_setup import logger
from prompts.prompt_abbrv import ASSISTANT_TEMPLATE, PROMPT_TEMPLATE

MAX_RETRIES_NVIDIA_CLIENT = 10

# ---------- Loggers ----------
batch_producer_logger = logging.getLogger("app_logger.batch_producer")
batch_consumer_logger = logging.getLogger("app_logger.batch_consumer")

# ---------- Sentinel ----------
STOP = object()

# ---------- Helpers ----------
def get_batches(data: List[Dict], batch_size: int) -> List[tuple[int, List[Dict]]]:
    """
    Split data into fixed-size batches and return a list of
    (batch_number, batch_items) tuples, where batch_number starts at 0.
    """
    return [
        (batch_num, data[i:i + batch_size])
        for batch_num, i in enumerate(range(0, len(data), batch_size))
    ]

def process_response(batch: str, resp, consumer_id: int) -> Optional[list[int]]:
    """Processing and validation of  API response"""
    log = logging.getLogger(f"app_logger.model_call.{consumer_id}")

    content = (
        resp.choices[0].message.content
        if (resp.choices and resp.choices[0].message and resp.choices[0].message.content)
        else ""
    )

    if not content:
        log.warning('API returned None content.')
        return None

    try:
        result_return = []
        results = json.loads(content)
        for result in results:
            if result.strip() != "":
                result_return.append(result)

        # text = " ".join(results)

        # if text != batch:
        #     log.warning(f"Hallucination!\nOriginal chunk: {repr(batch)}\nAPI response: {repr(results)}")

        return results
    
    except ValidationError:
        log.warning(f"Validation error: {ValidationError}")
        return None
    except Exception as e:
        log.error(f'Error happened while processing API response: {e}')
        return None

# ---------- Model call ----------
async def call_model(nv: NvidiaClient, batch: List[Dict], consumer_id: int) -> Optional[tuple]:
    log = logging.getLogger(f"app_logger.model_call.{consumer_id}")
    documents_str = json.dumps(batch, ensure_ascii=False)
    user_prompt = PROMPT_TEMPLATE.replace("{input_chunk}", documents_str)

    messages = [
        {"role": "system", "content": ASSISTANT_TEMPLATE},
        {"role": "user", "content": user_prompt},
    ]

    resp = await nv.chat_completion(messages=messages, reasoning_effort='medium')

    filtered_indexes = process_response(batch, resp, consumer_id)

    if isinstance(filtered_indexes, list):
        usage= {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        log.info(f'Batch has been filtered. Results:\n{usage}')
        
        if filtered_indexes:
            return (True, filtered_indexes)
        return (False, filtered_indexes)
    else:
        return None

# ---------- Stage 1: Batch producer ----------
async def batch_producer(batch_queue: asyncio.Queue, batches: List[List[tuple]]) -> None:
    batch_producer_logger.info("Batch Producer started.")
    for batch in batches:
        await batch_queue.put(batch)  # backpressure via maxsize
    batch_producer_logger.info("Producer finished enqueuing batches.")

# ---------- Stage 1: Batch consumers (produce results) ----------
async def batch_consumer(
    consumer_id: int,
    batch_queue: asyncio.Queue,
    nv: NvidiaClient,
    out_path: str
) -> None:
    log = logging.getLogger(f"app_logger.batch_consumer.{consumer_id}")
    log.info("Consumer started.")
    try:
        while True:
            # get batch from batch_queue
            item: list[tuple] = await batch_queue.get()
            log.info(f'Consumer_{consumer_id} got a batch of queries to process!')
            if item is STOP:
                batch_queue.task_done()
                log.info("Received STOP, exiting.")
                break

            batch_index = item[0]
            if (batch_index % 10000) == 0:
                log.info(f'Started batch: {batch_index}!')

            batch = item[1]
            try:
                # filter batch
                for _ in range(MAX_RETRIES_NVIDIA_CLIENT):
                    result = await call_model(nv, batch, consumer_id)
                    if result:
                        break

                if result[0]:
                    result_dict = {f'chunk_{batch_index+1}': result[1], 'original_chunk': batch}
                    await append_jsonl(out_path, result_dict)
                    log.info('Batch has been saved!')
                else:
                    log.info("Empty list returned!")

            except Exception:
                result_dict = {f'{batch_index}': None}
                await append_jsonl(out_path, result_dict)
                log.exception(f"Error processing batch. Batch index: {batch_index}")
            finally:
                batch_queue.task_done()
    except asyncio.CancelledError:
        log.info("Consumer cancelled.")
        raise
    finally:
        log.info("Consumer exited.")
        exit()

# ---------- Stage 2: Result writer ----------
async def append_jsonl(path: str, item: any) -> None:
    """Append a JSON-encoded item to a .jsonl file asynchronously."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    async with aiofiles.open(path, "a", encoding="utf-8") as f:
        await f.write(json.dumps(item, ensure_ascii=False) + "\n")
        await f.flush()

# ---------- Main ----------
async def main():
    logger.info("Application started.")
    logger.info("=== Application Configuration ===")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"MAX_RETRIES: {MAX_RETRIES}")
    logger.info(f"RETRY_DELAY (seconds): {RETRY_DELAY}")
    logger.info(f"MAX_CONCURRENT_BATCHES: {MAX_CONCURRENT_BATCHES}")
    logger.info("=== End of Configuration ===")

    # Load input JSONL
    data = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        block_data: dict = json.load(f)
        for block_key, block_value in block_data.items():
            for chunk_key, chunk_value in block_value.items():
                data.append(chunk_value)
            

    # Batch the data
    batches = get_batches(data, BATCH_SIZE)
    # random.shuffle(batches)
    logger.info("=== Batches have been generated ===")
    logger.info(f"Batch count: {len(batches)}")

    # Queues
    batch_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_CONCURRENT_BATCHES)

    # Client
    nv = NvidiaClient()
    print(batches[19190:19210])
    # Tasks
    producer_task = asyncio.create_task(batch_producer(batch_queue, batches[19190:19210]))
    consumers = [
        asyncio.create_task(batch_consumer(i + 1, batch_queue, nv, OUTPUT_PATH))
        for i in range(MAX_CONCURRENT_BATCHES)
    ]

    try:
        # Wait for producer to finish
        await producer_task

        # One STOP per consumer
        for _ in consumers:
            await batch_queue.put(STOP)

        # Wait until all batches are processed
        await batch_queue.join()

        # Ensure consumers & writer complete
        await asyncio.gather(*consumers)
        exit()
    except Exception:
        logger.exception("Fatal error in main")
        raise
    finally:
        try:
            await nv.close()
        except Exception:
            pass
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())