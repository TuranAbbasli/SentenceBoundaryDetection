
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from training import TextSegmenter


class SentenceSegmenter:
    """
    Handles segmenting text into sentences.
    """

    @staticmethod
    def segment_to_sentences(
        segmenter: "TextSegmenter",
        text: str,
        streaming: bool = False,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Segment text into a list of sentences.

        Args:
            segmenter: The TextSegmenter to use
            text (str): Text to segment
            streaming (bool, optional): Whether to use streaming mode for memory efficiency.
                                       Defaults to False.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[str]: List of sentences
        """
        # Quick return for empty text
        if not text:
            return []

        # Use optimized segmentation based on text size
        if streaming and len(text) > 10000:
            # For large texts, use streaming segmentation
            # Note: streaming mode doesn't currently support custom threshold
            segmented_parts = list(segmenter.segment_text_streaming(text))
            segmented_text = "".join(segmented_parts)
        else:
            # For smaller texts, use regular segmentation
            segmented_text = segmenter.segment_text(text, threshold=threshold)

        # Fast path: if no sentence tags were added, return the whole text as one sentence
        if segmenter.SENTENCE_TAG not in segmented_text:
            return [text] if text else []

        # More efficient string splitting and processing

        # Split by sentence tag, but handle paragraph tags properly
        sentences = []
        segments = segmented_text.split(segmenter.SENTENCE_TAG)

        # First segment is always before any sentence tag
        if segments[0]:
            sentences.append(segments[0])

        # Process remaining segments (each starts after a sentence tag)
        for segment in segments[1:]:
            # Remove any paragraph tags at the beginning of the segment

            if segment:
                sentences.append(segment)

        # Post-processing to fix incorrectly segmented quotation marks
        # This handles edge cases where the model fails to correctly process quotes
        i = 0
        while i < len(sentences) - 1:
            # Handle case where a sentence ends with a quote and next "sentence" is just a quote
            if (sentences[i].endswith('"') or sentences[i].endswith('"')) and sentences[
                i + 1
            ].strip() == '"':
                # Merge the quote with the following sentence
                if i + 2 < len(sentences):
                    sentences[i + 2] = '" ' + sentences[i + 2]
                    sentences.pop(i + 1)  # Remove the standalone quote
                    continue
            # Handle case where a "sentence" is just a quote that should connect to the next sentence
            if sentences[i].strip() == '"' and i + 1 < len(sentences):
                # Join with the next sentence
                sentences[i + 1] = '" ' + sentences[i + 1]
                sentences.pop(i)  # Remove the standalone quote
                continue
            i += 1

        return sentences