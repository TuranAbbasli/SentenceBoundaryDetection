#!/usr/bin/env python3
"""
Example script to demonstrate the usage of the CharBoundary library.
"""

import gzip
import json
import time
import random 
from constants import SENTENCE_TAG
from training import TextSegmenter
from models import ModelIO


def demonstrate_basic_usage():
    """Demonstrate basic usage of the CharBoundary library."""
    # Create a segmenter
    segmenter = TextSegmenter()
    
    # Load annotated training data (texts contain SENTENCE_TAG)
    training_data = []
    with gzip.open("data/train_data_v1.jsonl.gz", "rt", encoding="utf-8") as input_file:
        for i, line in enumerate(input_file):
            obj = json.loads(line)
            text = obj.get("text")
            if text:           # skip empty or missing lines
                training_data.append(text)
                if i>4:
                    break

    # Shuffle and split into train and test
    random.shuffle(training_data)
    split_ratio = 0.8
    split_idx = int(len(training_data) * split_ratio)
    train_texts = training_data[:split_idx]
    test_texts = training_data[split_idx:]

    print(f"Total samples: {len(training_data)}")
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples:  {len(test_texts)}")

    # Train the segmenter on the train part
    print("Training segmenter...")
    t0 = time.time()
    train_metrics = segmenter.train(
        data=train_texts,                         # train only
        model_params={"n_estimators": 100, "max_depth": 16},
        sample_rate=0.001,
        left_window=9,
        right_window=9,
    )
    print("Training completed in {:.2f} seconds.".format(time.time() - t0))
    print("Training metrics:", train_metrics)

    # Evaluate on the held out test part
    print("\nEvaluating on test set...")
    test_metrics = segmenter.evaluate(
        data=test_texts,                          
        max_samples=None,
    )
    print("Test metrics:", test_metrics)

    # Save the trained model
    ModelIO.save(
        segmenter=segmenter,
        path="models/charboundary_segmenter.skops",
        format="skops",
        compress=True,
        compression_level=9,
    )
    print("Model saved to models/charboundary_segmenter.skops.xz")
    
    # Example texts to segment (raw text, no SENTENCE_TAG here)
    examples = [
        "2.1. Ölkənin sosial-iqtisadi, mədəni inkişafının ümummilli strateji siyasətinə uyğun şəkildə ____________ rayon (şəhər) gənclərinin tərbiyəsi aparılır. Maddə 18. Demokratik prinsiplərə, ümumbəşəri və milli dəyərlərə yiyələnməsinə kömək göstərməklə 3 trln. gənclərin qüvvə və bacarığını, yaradıcı potensialını dövlət quruculuğuna onun suverenliyinin möhkəmləndirilməsinə, ____________________ rayonun (şəhərin) iqtisadi və sosial inkişafı məsələlərinin həllinə səfərbərdir. Xahiş olunur ki, bu prosesdə gənclərin fəal iştirakı təmin edilsin. Həmçinin 1.4.2 maddəsinə əsasən, gənclərin dövlət orqanlarında, bələdiyyələrdə və qeyri-hökumət təşkilatlarında ictimai fəallığının artırılması məqsədilə onların hüquq və imkanlarının genişləndirilməsi təmin edilsin.",
        "2.1. Ölkənin sosial-iqtisadi, mədəni inkişafının ümummilli strateji siyasətinə uyğun şəkildə ____ rayon (şəhər) gənclərinin tərbiyəsi aparılır. Maddə 18. Demokratik prinsiplərə, ümumbəşəri və milli dəyərlərə yiyələnməsinə kömək göstərməklə 3 trln. gənclərin qüvvə və bacarığını, yaradıcı potensialını dövlət quruculuğuna onun suverenliyinin möhkəmləndirilməsinə, ________ rayonun (şəhərin) iqtisadi və sosial inkişafı məsələlərinin həllinə səfərbərdir. Xahiş olunur ki, bu prosesdə gənclərin fəal iştirakı təmin edilsin. Həmçinin 1.4.2 maddəsinə əsasən, gənclərin dövlət orqanlarında, bələdiyyələrdə və qeyri-hökumət təşkilatlarında ictimai fəallığının artırılması məqsədilə onların hüquq və imkanlarının genişləndirilməsi təmin edilsin."
    ]
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Original: {example}")
        
        sentences = segmenter.segment_to_sentences(example)
        print("\nSentences:")
        for j, sentence in enumerate(sentences):
            print(f"  {j+1}. {sentence}")


def main():
    """Run the example script."""
    print("CharBoundary Library Example")
    demonstrate_basic_usage()


if __name__ == "__main__":
    main()
