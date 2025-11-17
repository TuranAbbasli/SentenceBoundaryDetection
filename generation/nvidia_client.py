# -*- coding: utf-8 -*-
"""
Async NVIDIA GPT-OSS client built on the OpenAI Python SDK.

- Creates one AsyncOpenAI client per API key and cycles through them.
- Retries transient errors with exponential backoff.
- Accepts iterables for messages and minimal kwargs for behavior.

Usage example:

    client = NvidiaClient()
    resp = await client.chat_completion(
        messages=[{"role":"user","content":"Hello"}],
        reasoning_effort="medium",
        extra_body={"nvext": {"guided_json": MySchema.model_json_schema()}}
    )

    await client.close()  # gracefully close pooled clients when your app exits
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import random
from typing import Any, Dict, Iterable, Optional, Tuple

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

import config2
from logging_setup_yusif import *

setup_logging()

class ConnectionsRefused(Exception):
    ...


log = logging.getLogger(__name__)

key_log_slice = slice(len("nvapi-"), len("nvapi-") + 6)


class NvidiaClient:
    """
    Lightweight wrapper around OpenAI Async client targeting NVIDIA's base_url.
    Maintains a cycle of AsyncOpenAI clients.
    """

    def __init__(self):
        if not config2.API_KEYS:
            raise ValueError("No API keys set in config2.API_KEYS")

        keys = config2.API_KEYS if isinstance(
            config2.API_KEYS, (list, tuple)) else [config2.API_KEYS]
        print("DEBUG: BASE_URL is", config2.BASE_URL)
        clients = [AsyncOpenAI(
            api_key=k, base_url=config2.BASE_URL.rstrip("/")) for k in keys]
        self._client_cycle = itertools.cycle(clients)
        self.model = config2.MODEL_ID

        for k in keys:
            log.info("Async client created", extra={
                     "base_url": config2.BASE_URL, "api_key_suffix": k[-4:]})

    async def close(self) -> None:
        for cli in set(self._client_cycle):
            try:

                await cli.close()
                log.info("Closing client", extra={
                         "api_key": cli.api_key[key_log_slice]})

            except Exception:
                pass

    async def chat_completion(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> ChatCompletion:
        """
        Perform a chat completion using a pooled AsyncOpenAI client.
        """
        msgs: Tuple[Dict[str, str], ...] = tuple(messages)

        retries = 0
        max_retries = max_retries or config2.MAX_RETRIES
        backoff = config2.INITIAL_BACKOFF
    
        while (retries := retries + 1) <= max_retries:
            client = next(self._client_cycle)
            api_key = client.api_key

            try:
                log.info("Request sent to API!")
                completion = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=list(msgs),
                    reasoning_effort=reasoning_effort or config2.REASONING_EFFORT,
                    temperature=config2.TEMPERATURE if temperature is None else temperature,
                    top_p=config2.TOP_P if top_p is None else top_p,
                    max_tokens=config2.MAX_TOKENS if max_tokens is None else max_tokens,
                    stream=False,
                    extra_body={
                                "nvext": {
                                    "guided_json": config2.llm_response_schema
                                }
                            }
                )

                if completion.choices is None:
                    raise ConnectionsRefused(completion.error)

                return completion

            except Exception as e:
                should_retry = (
                    isinstance(e, (openai.RateLimitError, ConnectionsRefused,
                               openai.APITimeoutError, openai.APIConnectionError))
                    or (isinstance(e, openai.APIError) and getattr(e, "status", 500) >= 500)
                )
                if not should_retry:
                    log.error(
                        "API call failed permanently",
                        extra={"error": str(
                            e), "retries": retries, "api_key": api_key[key_log_slice]},
                    )
                    raise

                log.warning(
                    "API call failed; retrying",
                    extra={"error": str(e), "retry": retries,
                           "api_key": api_key[key_log_slice]},
                )
                await asyncio.sleep(backoff + random.random())
                backoff = min(
                    backoff * 2, getattr(config2, "MAX_BACKOFF", 30.0))

        raise RuntimeError(
            f"Max retries exceeded ({max_retries}) for API key {api_key[:6]}...")


async def main():
    
    client = NvidiaClient()
    for _ in range(5):
        resp = await client.chat_completion(messages=[
            {"role":"system", "content":"YOU TELL ONLY THE TRUTH"},
            {"role":"user", "content":"hello?"}])
        
        print(resp)


if __name__ == "__main__":
    asyncio.run(main())
