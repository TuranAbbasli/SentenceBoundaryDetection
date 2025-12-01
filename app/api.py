import logging
from contextlib import asynccontextmanager

import tritonclient.grpc as grpcclient
from fastapi import FastAPI, HTTPException

from .config import TRITON_URL, MODEL_NAME, NETWORK_TIMEOUT
from .models import SegmentationResponse
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


@app.post("/segment", response_model=SegmentationResponse)
async def segment_text(text: str):
    """
    Segment input text into sentences.
    
    Returns the text with SENTENCE_TAG markers inserted at predicted boundaries.
    """
    triton_client = get_triton_client()
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Triton client not initialized")
    
    try:
        # Preprocessing
        inputs, outputs, positions = preprocessing(text, left_window=9, right_window=9)

        # Inference
        if inputs is None or outputs is None or not positions:
            # No terminal characters found
            return SegmentationResponse(
                original_text=text,
                segmented_text=text,
                num_boundaries=0,
            )

        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            client_timeout=NETWORK_TIMEOUT,
            headers={},
        )

        # Postprocessing
        segmented_text, boundary_count = postprocessing(text, positions, response)

        return SegmentationResponse(
            original_text=text,
            segmented_text=segmented_text,
            num_boundaries=boundary_count,
        )

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
