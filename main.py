#!/usr/bin/env python3
"""
Example script to demonstrate training and evaluation of a TextSegmenter
from the azcharboundary library.
"""

import json
import time
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List

from azcharboundary.segmenter import TextSegmenter
from azcharboundary.utils.types import MetricsResult

def load_dataset(data_path: Path) -> List[str]:
    """
    Load a JSONL dataset from disk.

    Each line is expected to be a JSON object with an "input" field.

    Args:
        data_path (Path): Path to the JSONL data file.

    Returns:
        list[str]: List of input texts.
    """
    print("Loading data...")
    with data_path.open("r", encoding="utf-8", errors="replace") as f:
        records: List[dict] = [json.loads(line) for line in f]

    texts = [item["input"] for item in records]
    print(f"Loaded {len(texts)} samples.")
    return texts


def train_test_split(
    data: Sequence[str],
    test_ratio: float = 0.2,
    shuffle: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Split data into train and test sets.

    Args:
        data (Sequence[str]): Full dataset.
        test_ratio (float): Proportion of the dataset to allocate to the test set.
        shuffle (bool): Whether to shuffle data before splitting.

    Returns:
        tuple[list[str], list[str]]: (train_set, test_set)
    """
    data_list = list(data)

    if shuffle:
        random.shuffle(data_list)

    data_len = len(data_list)
    split_idx = int(data_len * test_ratio)

    test_set: List[str] = data_list[:split_idx]
    train_set: List[str] = data_list[split_idx:]

    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")
    return train_set, test_set


def print_metrics(metrics: MetricsResult, name: str = "Process") -> None:
    """
    Pretty-print standard segmentation metrics.

    Args:
        metrics (MetricsResult): Metrics dictionary produced by the segmenter.
        name (str): Name of the process (e.g., "Training", "Evaluation").
    """
    print(f"\n{name} metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0.0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0.0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0.0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0.0):.4f}")


def save_model(
    segmenter: TextSegmenter,
    path: str = "./",
    serialization_format: str = "treelite",
) -> None:
    """
    Save a trained segmenter model to disk.

    Args:
        segmenter (TextSegmenter): Trained segmenter instance.
        path (str): Path where the model will be saved.
        serialization_format (str): Serialization format (e.g. "treelite").
    """
    save_start = time.time()
    segmenter.save(path=path, serialization_format=serialization_format)
    save_end = time.time()
    print("\nTime taken to save model: {:.2f} seconds".format(save_end - save_start))


def train_segmenter(segmenter: TextSegmenter, train_set: Iterable[str]) -> MetricsResult:
    """
    Train the TextSegmenter on the provided training set.

    Args:
        segmenter (TextSegmenter): Segmenter instance to train.
        train_set (Iterable[str]): Training texts.

    Returns:
        MetricsResult: Training metrics.
    """
    train_set = list(train_set)
    print(f"Training segmenter with {len(train_set)} training samples.\n")

    train_start = time.time()
    training_metrics = segmenter.train(
        data=train_set,
        model_params={"n_estimators": 128, "max_depth": 32},
        sample_rate=0.001,          # Increase sample rate to get better class balance
        left_window=5,              # Specify window sizes during training
        right_window=5,
        threshold=0.8,
        use_feature_selection=False,
        feature_selection_threshold=0.01,
        max_features=20,
    )
    print("Training completed in {:.2f} seconds.".format(time.time() - train_start))
    print_metrics(metrics=training_metrics, name="Training")
    return training_metrics


def evaluate_segmenter(segmenter: TextSegmenter, evaluation_set: Iterable[str]) -> MetricsResult:
    """
    Evaluate the TextSegmenter on the provided evaluation set.

    Args:
        segmenter (TextSegmenter): Trained segmenter instance.
        evaluation_set (Iterable[str]): Evaluation texts.

    Returns:
        MetricsResult: Evaluation metrics.
    """
    evaluation_set = list(evaluation_set)
    print(f"\nEvaluating on test set of size {len(evaluation_set)}.")

    evaluation_start = time.time()
    evaluation_metrics = segmenter.evaluate(
        data=evaluation_set,
        max_samples=None,
        sample_rate=0.001,
    )
    print("Evaluation completed in {:.2f} seconds.".format(time.time() - evaluation_start))
    print_metrics(metrics=evaluation_metrics, name="Evaluation")
    return evaluation_metrics


def main(data_path: Path, save_path: str = "./") -> None:
    """
    Run the full training and evaluation pipeline.

    Steps:
        1. Load dataset.
        2. Split into train and test sets.
        3. Train segmenter.
        4. Evaluate segmenter.
        5. Save trained model.

    Args:
        data_path (Path): Path to the JSONL data file.
        save_path (str): Path where the trained model will be saved.
    """
    # Initialize segmenter
    segmenter = TextSegmenter()

    # Load & split data
    dataset = load_dataset(data_path=data_path)
    train_set, test_set = train_test_split(dataset, test_ratio=0.1)

    # Train & evaluate
    train_segmenter(segmenter=segmenter, train_set=train_set)
    evaluate_segmenter(segmenter=segmenter, evaluation_set=test_set)

    # Save model
    save_model(segmenter=segmenter, path=save_path, serialization_format="treelite")


if __name__ == "__main__":
    data_path = Path("azcharboundary/data/final.jsonl")
    save_path = "azcharboundary/trained_models/v3/checkpoint.tl"

    main(data_path, save_path)
