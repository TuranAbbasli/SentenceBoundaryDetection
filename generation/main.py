# main.py
"""
Main training script for SBD data generation
"""
import asyncio
import json
import logging
import os
import random
from typing import Optional
import aiofiles
from pydantic import ValidationError

from config import (
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_CONCURRENT_BATCHES,
    INPUT_PATH,
    OUTPUT_PATH
)
import model
from nvidia_client import NvidiaClient
from logging_setup import logger
from prompts.prompt_sbd import ASSISTANT_TEMPLATE, PROMPT_TEMPLATE

MAX_RETRIES_NVIDIA_CLIENT = 10

# ---------- Loggers ----------
chunk_producer_logger = logging.getLogger("app_logger.chunk_producer")
chunk_consumer_logger = logging.getLogger("app_logger.chunk_consumer")

# ---------- Sentinel ----------
STOP = object()

def process_response(resp, consumer_id: int) -> Optional[list[int]]:
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
        results = json.loads(content)

        validated_results = model.Sbd_Output(results=results)
        return validated_results.results
    
    except ValidationError:
        log.warning(f"Validation error: {ValidationError}")
        return None
    except Exception as e:
        log.error(f'Error happened while processing API response: {e}')
        return None

# ---------- Model call ----------
async def call_model(nv: NvidiaClient, chunk: str, consumer_id: int) -> Optional[tuple]:
    log = logging.getLogger(f"app_logger.model_call.{consumer_id}")
    documents_str = json.dumps(chunk, ensure_ascii=False)
    user_prompt = PROMPT_TEMPLATE.replace("{input_chunk}", documents_str)

    messages = [
        {"role": "system", "content": ASSISTANT_TEMPLATE},
        {"role": "user", "content": user_prompt},
    ]

    resp = await nv.chat_completion(messages=messages, reasoning_effort='medium')

    filtered_indexes = process_response(resp, consumer_id)

    if isinstance(filtered_indexes, list):
        usage= {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        log.info(f'chunk has been filtered. Results:\n{usage}')
        
        return filtered_indexes
    else:
        return None

def get_processed_chunk_indexes() -> list[str]:
    """Returns indexes of processed chunks which are saved in OUTPUT_PATH"""
    chunk_idxs: list[str] = []

    chunk_producer_logger.info("started getting indexes of processed chunks.")
    with open(OUTPUT_PATH, "r", encoding="utf-8", errors="replace") as f_processed:
        for i, line in enumerate(f_processed, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {i}: {e}")
                continue
            
            for key in data.keys():
                if key.startswith("chunk_"):
                    chunk_idxs.append(key)
    chunk_producer_logger.info(f"finished getting indexes of processed chunks. Count: {len(chunk_idxs)}")

    return chunk_idxs

# ---------- Stage 1: Chunk producer ----------
async def chunk_producer(chunk_queue: asyncio.Queue, chunks: list[tuple]) -> None:
    processed_idxs = get_processed_chunk_indexes()
    enqueued_count = 0

    chunk_producer_logger.info("chunk Producer started.")
    for chunk in chunks:
        if chunk[0] not in processed_idxs:
            await chunk_queue.put(chunk)  # backpressure via maxsize
            enqueued_count += 1
    chunk_producer_logger.info(f"Producer finished enqueuing chunkes.")

# ---------- Stage 1: Chunk consumers (produce results) ----------
async def chunk_consumer(
    consumer_id: int,
    chunk_queue: asyncio.Queue,
    nv: NvidiaClient,
    out_path: str
) -> None:
    log = logging.getLogger(f"app_logger.chunk_consumer.{consumer_id}")
    log.info("Consumer started.")
    try:
        while True:
            # get chunk from chunk_queue
            item: list[tuple] = await chunk_queue.get()
            log.info(f'Consumer_{consumer_id} got a chunk of queries to process!')
            if item is STOP:
                chunk_queue.task_done()
                log.info("Received STOP, exiting.")
                break

            chunk_index = item[0]
            chunk_value = item[1]
            try:
                # filter chunk
                for _ in range(MAX_RETRIES_NVIDIA_CLIENT):
                    result = await call_model(nv, chunk_value, consumer_id)
                    if result:
                        break

                result_dict = {f'chunk_{chunk_index}': result, 'original_chunk': chunk_value}
                await append_jsonl(out_path, result_dict)
                log.info('chunk has been saved!')

            except Exception:
                result_dict = {f'{chunk_index}': None}
                await append_jsonl(out_path, result_dict)
                log.exception(f"Error processing chunk. chunk index: {chunk_index}")
            finally:
                chunk_queue.task_done()
    except asyncio.CancelledError:
        log.info("Consumer cancelled.")
        raise
    finally:
        log.info("Consumer exited.")

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
    logger.info(f"MAX_RETRIES: {MAX_RETRIES}")
    logger.info(f"RETRY_DELAY (seconds): {RETRY_DELAY}")
    logger.info(f"MAX_CONCURRENT_CHUNKS: {MAX_CONCURRENT_BATCHES}")
    logger.info("=== End of Configuration ===")

    # Load input JSONL
    chunks: list[tuple] = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        block_data: dict = json.load(f)
        for block_key, block_value in block_data.items():
            for chunk_key, chunk_value in block_value.items():
                chunks.append((chunk_key, chunk_value))    

    # random.shuffle(data)
    logger.info("=== Chunks have been generated ===")
    logger.info(f"chunk count: {len(chunks)}")

    # Queues
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_CONCURRENT_BATCHES)

    # Client
    nv = NvidiaClient()

    # Tasks
    producer_task = asyncio.create_task(chunk_producer(chunk_queue, chunks))
    consumers = [
        asyncio.create_task(chunk_consumer(i + 1, chunk_queue, nv, OUTPUT_PATH))
        for i in range(MAX_CONCURRENT_BATCHES)
    ]

    try:
        # Wait for producer to finish
        await producer_task

        # One STOP per consumer
        for _ in consumers:
            await chunk_queue.put(STOP)

        # Wait until all chunkes are processed
        await chunk_queue.join()

        # Ensure consumers & writer complete
        await asyncio.gather(*consumers)
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