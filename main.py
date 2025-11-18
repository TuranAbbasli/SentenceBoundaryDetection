#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library.
"""

import json
import time
from pathlib import Path

from azcharboundary.segmenter import TextSegmenter

def demonstrate_basic_usage(data_dir: Path):
    """Demonstrate basic usage of the CharBoundary library."""
    # Create a segmenter
    segmenter = TextSegmenter()
    
    # Sample annotated text for training

    with open(data_dir, "r", encoding="utf-8", errors="replace") as f:
        preprocessed_data = [json.loads(line) for line in f]

    training_data = [item["output"] for item in preprocessed_data]

    # Train the segmenter
    print("Training segmenter...")
    t0 = time.time()
    metrics = segmenter.train(
        data=training_data,
        model_params={"n_estimators": 512, "max_depth": 64},
        sample_rate=0.001,  # Increase sample rate to get better class balance
        left_window=9,  # Specify window sizes during training
        right_window=9
    )
    print("Training completed in {:.2f} seconds.".format(time.time() - t0))
    
    # Display training metrics
    print(f"Training metrics:")
    print(f"  Overall accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Boundary accuracy:      {metrics.get('boundary_accuracy', 0):.4f}")
    print(f"  Boundary precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Boundary recall:        {metrics.get('recall', 0):.4f}")
    print(f"  Boundary F1-score:      {metrics.get('f1_score', 0):.4f}")

def main():
    """Run the example script."""
    print("CharBoundary Library Example\n")
    data_dir = Path(r'azcharboundary\data\data.jsonl')
    demonstrate_basic_usage(data_dir)


if __name__ == "__main__":
    main()
