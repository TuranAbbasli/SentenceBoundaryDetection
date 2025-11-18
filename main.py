#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library.
"""

import json
import time
from pathlib import Path

from azcharboundary.segmenter import TextSegmenter
from azcharboundary.utils.constants import (SENTENCE_TAG,
                                            CONFUSABLE_MAP,
                                            ALLOWED_AZ_CHARS)

def preprocessing(data_dir: Path) -> list[dict]:
    """
    Loads generation results. Current version further splits sentences by '\n'.
    
    Args:
        data_dir (Path): path to generation results

    Returns:
        list[dict]: list of input-output chunks
    """

    # training data with input-output values
    preprocessed_data: list[dict] = [] 

    with open(data_dir, "r", encoding="utf-8", errors="replace") as f_raw:
        for i, line in enumerate(f_raw, start=1):
            line = line.strip()

            if not line:  # skip emtpy lines
                print(f'Empty line in raw file: {i}')
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {i}: {e}")
                continue

            sentences: list[str] = next(iter(data.values())) # first value from key-value pairs

            if not sentences:
                print(f'Empty value in generation data on line: {i}.')
                continue
            
            updated_sentences: list[str] = []
            for sentence in sentences:  # further splitting sentences of a chunk
                for s in sentence.split('\n'):
                    s = s.strip()
                    s = ''.join(CONFUSABLE_MAP.get(char, char) for char in s)  # fix confusables

                    # check CharValidness of sentence
                    valid_sentence = True
                    for char in s:
                        if char.isalpha():
                            if char not in ALLOWED_AZ_CHARS:
                                valid_sentence = False

                    if valid_sentence:  # skip non-valid sentence
                        updated_sentences.append(s)

            training_data = {
                "input": " ".join(updated_sentences),
                "output": f"{SENTENCE_TAG} ".join(updated_sentences)
            }
            preprocessed_data.append(training_data)

        return preprocessed_data

def demonstrate_basic_usage(data_dir: Path):
    """Demonstrate basic usage of the CharBoundary library."""
    # Create a segmenter
    segmenter = TextSegmenter()
    
    # Sample annotated text for training

    preprocessed_data = preprocessing(data_dir)
    training_data = [item["output"] for item in preprocessed_data]
    print(training_data[:5])
    exit()
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
    data_dir = Path(r'generation\results.jsonl')
    demonstrate_basic_usage(data_dir)


if __name__ == "__main__":
    main()
