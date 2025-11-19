#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library.
"""

import json
import time
from pathlib import Path

from azcharboundary.segmenter import TextSegmenter

def train_test_split(data: list, ratio: int = 0.2) -> tuple[list, list]:
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

    test_set: list = data[:splitting_idx]
    train_set: list = data[splitting_idx:]

    return train_set, test_set

def demonstrate_basic_usage(data_dir: Path, save_dir: str): 
    """Demonstrate basic usage of the CharBoundary library."""
    # Create a segmenter
    segmenter = TextSegmenter()
    
    # Sample annotated text for training

    with open(data_dir, "r", encoding="utf-8", errors="replace") as f:
        preprocessed_data = [json.loads(line) for line in f]

    datas = [item["text"] for item in preprocessed_data]
    train_set, test_set = train_test_split(datas)

    # Train the segmenter
    print(f"Training segmenter with {len(train_set)} training data.")
    train_start = time.time()
    metrics = segmenter.train(
        data=train_set,
        model_params={"n_estimators": 512, "max_depth": 64},
        sample_rate=0.001,  # Increase sample rate to get better class balance
        left_window=5,      # Specify window sizes during training
        right_window=5
    )
    print("Training completed in {:.2f} seconds.".format(time.time() - train_start))
    
    # Display training metrics
    print(f"Training metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")

    print(f"Evaluation on test set sized {len(test_set)}.")
    evaluation_start = time.time()
    evaluation_metrics = segmenter.evaluate(
        data=test_set,                          
        max_samples=None,
    )

    # Display evaluation metrics
    print(f"Evaluation metrics:")
    print(f"  Overall accuracy:       {evaluation_metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {evaluation_metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {evaluation_metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {evaluation_metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {evaluation_metrics.get('f1_score', 0):.4f}")
    print("Evaluation completed in {:.2f} seconds.".format(time.time() - evaluation_start))
    

    save_start = time.time()
    segmenter.save(path=save_dir)
    save_end = time.time()
    print('Time took to save model: {:.2f}'.format(save_end-save_start))


def main():
    """Run the example script."""
    data_dir = Path(r'azcharboundary\data\train_data_v3_fixed.jsonl')
    save_dir = "azcharboundary/models/model_v1.xz"

    demonstrate_basic_usage(data_dir, save_dir)


if __name__ == "__main__":
    main()
