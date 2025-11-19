"""
Example script to demonstrate using a *saved* CharBoundary model.
"""
import time
import sys
from training import TextSegmenter
from models import ModelIO


def demonstrate_loaded_model():
    """Load a previously saved model and use it to segment text."""
    print("CharBoundary Library Example (using saved model)")

    # Load the saved model
    segmenter = ModelIO.load(
        path="models/charboundary_segmenter.skops",
        segmenter_class=TextSegmenter,
        use_skops=True,
        trust_model=True,  # set to False if you want skops to be strict
    )

    # Example texts to segment
    examples = [
        "Qeyd: Bu Məcəllənin 513.1 və 513.2-ci maddələrində nəzərdə tutulmuş əməllərdə cinayət tərkibinin əlamətləri olduqda, həmin əməllər Azərbaycan Respublikası Cinayət Məcəlləsinin müvafiq maddələrinə əsasən cinayət məsuliyyətinə səbəb olur.",
        "Maddə 216. Bu onu göstərir ki XX. cümlədə problem var. Həmçinin 300 man. məbləğində paltar aldıq."
    ]

    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Original:\n{example}\n")
        start = time.time()
        sentences = segmenter.segment_to_sentences(example)
        end = time.time()
        print("Sentences:")
        for j, sentence in enumerate(sentences):
            print(f"  {j+1}. {sentence}")
        print(f"Inference time: {end - start:.6f} seconds")


def main():
    """Entry point that only loads and uses the saved model."""
    try:
        demonstrate_loaded_model()
    except FileNotFoundError as e:
        print("Model file not found.")
        print("Make sure you have already trained and saved the model,")
        print("for example with your training script that calls ModelIO.save.")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
