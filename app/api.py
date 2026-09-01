import logging
from contextlib import asynccontextmanager

import tritonclient.grpc as grpcclient
from fastapi import FastAPI, HTTPException

from .config import TRITON_URL, MODEL_NAME, NETWORK_TIMEOUT
from .models import SegmentationResponse, SegmentationRequest
from .services import (
    get_triton_client,
    set_triton_client,
    warmup_model,
    preprocessing,
    postprocessing,
)

logger = logging.getLogger("AZSBDService")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info(f"Connecting to Triton server at {TRITON_URL}...")
    
    try:
        client = grpcclient.InferenceServerClient(
            url=TRITON_URL,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None,
        )
        
        # Workaround for older tritonclient
        if not hasattr(client, "_stream"):
            client._stream = None
        
        # Verify connection
        if not client.is_server_live():
            raise RuntimeError("Triton server is not live")
        if not client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton")
        
        logger.info("Connected to Triton server. Model is ready.")
        
        # Set global client
        set_triton_client(client)
        
        # Warm up the model
        warmup_model(client, num_requests=5)
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {e}")
        logger.warning("API will start but segmentation endpoint will not work until Triton is available")
    
    yield
    
    # Shutdown
    triton_client = get_triton_client()
    if triton_client:
        logger.info("Closing Triton client connection...")
        triton_client.close()
        logger.info("Connection closed.")


app = FastAPI(
    title="Azerbaijani Sentence Boundary Detection API",
    description="FastAPI service for sentence segmentation using Triton Inference Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """
    Liveness probe: confirms the process is answering requests.

    Does not touch Triton, so it stays green during a Triton outage.

    Returns:
        dict[str, str]: Static status payload.
    """
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """
    Readiness probe: confirms Triton is reachable and the model is loaded.

    Returns:
        dict[str, str]: Status, model name, and Triton URL when servable.

    Raises:
        HTTPException: 503 if the Triton client is missing, Triton is
                       unreachable, or the model is not ready.
    """
    client = get_triton_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Triton client not initialized")

    try:
        model_ready = client.is_model_ready(MODEL_NAME)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Triton unreachable: {e}")

    if not model_ready:
        raise HTTPException(status_code=503, detail=f"Model '{MODEL_NAME}' is not ready")

    return {"status": "ready", "model": MODEL_NAME, "triton": TRITON_URL}


@app.post("/segment", response_model=SegmentationResponse)
async def segment_text(request: SegmentationRequest):
    """
    Segment input text into sentences.
    
    Returns a list of detected sentences.
    """
    text = request.text
    triton_client = get_triton_client()
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client not initialized")
    
    try:
        # Preprocessing
        inputs, outputs, positions = preprocessing(text, left_window=9, right_window=9)

        # Inference
        if inputs is None or outputs is None or not positions:
            # No terminal characters found
            sentences = [text.strip()] if text.strip() else []
            return SegmentationResponse(
                sentences=sentences,
                num_sentences=len(sentences),
            )

        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            client_timeout=NETWORK_TIMEOUT,
            headers={},
        )

        # Postprocessing
        sentences = postprocessing(text, positions, response)

        return SegmentationResponse(
            sentences=sentences,
            num_sentences=len(sentences),
        )

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
