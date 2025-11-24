#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library.
"""

import json
import time
import random
from pathlib import Path

from azcharboundary.segmenter import TextSegmenter

def load_data(data_dir: Path) -> list[str]:
    """
    Loads data
    
    Args:
        data_dir (Path): path to data

    Returns:
        (list[str]): list of training datas
    """
    print("Loading data!")
    with open(data_dir, "r", encoding="utf-8", errors="replace") as f:
        preprocessed_data: list[dict] = [json.loads(line) for line in f]

    datas = [item["input"] for item in preprocessed_data]

    return datas

def train_test_split(data: list, ratio: int = 0.2, shuffle: bool = True) -> tuple[list, list]:
    """
    Splits loaded data into train and test sets.

    Args:
        data (list): list of loaded data
        ratio (int): ratio of train set over whole set

    Returns:
        tuple[list, list]: tuple where first item is train_set andn second is test_set
    """
    data_len = len(data)
    splitting_idx = int(data_len * ratio)

    if shuffle:
        random.shuffle(data)
        
    test_set: list = data[:splitting_idx]
    train_set: list = data[splitting_idx:]

    return train_set, test_set

def print_metrics(metrics: dict, name: str = "Process") -> None:
    """
    Prints metrics
    
    Args:
        metrics (dict): Metrics
        name (str): name of process metrics belong to. Default is "Process"
    """
    # Display training metrics
    print(f"\n{name} metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")

def save_model(segmenter: TextSegmenter, path: str = "./", serialization_format: str = "treelite") -> None:
    """
    Saves model.

    Args:
        segmenter (TextSegmenter): segmenter instance
        path (str): path to model. Default to "./
        serelization_format (str): format of serialization. Default to "treelite"   
    """
    save_start = time.time()
    segmenter.save(path=path, serialization_format=serialization_format)
    save_end = time.time()
    print('\nTime took to save model: {:.2f}'.format(save_end-save_start))

def demonstrate_basic_usage(data_dir: Path, save_dir: str): 
    """Demonstrate basic usage of the CharBoundary library."""
    # Create a segmenter
    segmenter = TextSegmenter()
    
    datas = load_data(data_dir=data_dir)
    train_set, test_set = train_test_split(datas, ratio=0.1)

    # Train the segmenter
    print(f"Training segmenter with {len(train_set)} training data.\n")

    train_start = time.time()
    training_metrics = segmenter.train(
        data=train_set,
        model_params={"n_estimators": 128, "max_depth": 32},
        sample_rate=0.001,    # Increase sample rate to get better class balance
        left_window=9,        # Specify window sizes during training
        right_window=9,
        threshold=0.5,
        use_feature_selection=False,
        feature_selection_threshold=0.01,
    )
    print("Training completed in {:.2f} seconds.".format(time.time() - train_start))
    print_metrics(metrics=training_metrics, name="Training")
    
    print(f"\nEvaluation on test set sized {len(test_set)}.")
    evaluation_start = time.time()
    evaluation_metrics = segmenter.evaluate(
        data=test_set,                          
        max_samples=None,
        sample_rate=0.001
    )
    print("Evaluation completed in {:.2f} seconds.".format(time.time() - evaluation_start))
    print_metrics(metrics=evaluation_metrics, name="Evaluation")

    save_model(segmenter=segmenter, path=save_dir, serialization_format="treelite")

def main():
    """Run the example script."""
    data_dir = Path(r'azcharboundary\data\train_data_v3_fixed.jsonl')
    save_dir = "azcharboundary/models/model_v1.tl"

    linux_data_dir = Path("azcharboundary/data/final.jsonl")
    linux_save_dir = "azcharboundary/trained_models/v3/checkpoint.tl"

    demonstrate_basic_usage(linux_data_dir, linux_save_dir)


if __name__ == "__main__":
    main()
