# Sentence Boundary Detection

This project implements a custom Sentence Boundary Detection system using the structure of the charboundary library.

## Training

Training is handled through main.py. You can adjust hyperparameters directly inside the training call:

    training_metrics = segmenter.train(
        data=train_set,
        model_params={"n_estimators": 128, "max_depth": 32},
        sample_rate=0.001,
        left_window=9,
        right_window=9,
        threshold=0.8,
        use_feature_selection=False,
        max_features=20,
    )

Modify these parameters to train a new model. Window sizes, thresholds, sample rate, and model configuration all impact performance.

## Inference with NVIDIA Triton

Inference is served using NVIDIA Triton Inference Server, which accelerates execution with GPU or CPU support. Docker Compose is used to run the server.

### Folder Structure

Your model should be placed inside a directory named model:

    triton_inference_server/
    ├─ docker-compose.yml
    └─ model/
       ├─ config.pbtxt
       └─ [model files]

The compose file mounts this directory as the Triton model repository.

### Starting the Server

From inside the Triton project directory, run:

    docker compose up -d 

The Triton server will start and load your SBD model for inference.
