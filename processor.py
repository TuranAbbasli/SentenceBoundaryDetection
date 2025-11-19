import json
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clean_sentence(text: str) -> str:
    """
    Remove newlines, collapse spaces, and trim whitespace.

    Args:
        text: Raw sentence text.

    Returns:
        Cleaned sentence.
    """
    if not text:
        return ""

    text = re.sub(r"(\\n|\n)+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def has_terminal(text: str) -> bool:
    """
    Check if a sentence ends with typical terminal punctuation.

    Args:
        text: Input string.

    Returns:
        True if text ends with ., !, ?, or …
    """
    return text.endswith((".", "!", "?", "…"))


def capitalize_first_letter(sentence: str) -> str:
    """
    Capitalize the first alphabetic letter of a sentence, if lowercase.

    Args:
        sentence: Input text.

    Returns:
        Sentence with first alphabetic character capitalized if needed.
    """
    if not sentence:
        return sentence

    for i, ch in enumerate(sentence):
        if ch.isalpha():
            if ch.islower():
                return sentence[:i] + ch.upper() + sentence[i + 1 :]
            break

    return sentence


def get_overlap(second_sentence: str) -> Optional[str]:
    """
    Extract usable overlap from the second sentence by removing the last word.

    Args:
        second_sentence: Sentence with <|sentence|> tag.

    Returns:
        Overlap string (all words except last), or None if too short.
    """
    clean_text = second_sentence.replace("<|sentence|>", "").strip()
    words = re.split(r"\s+", clean_text)

    if len(words) <= 2:
        return None

    return " ".join(words[:-1])

def load_and_clean_raw_data(input_path: str) -> List[str]:
    """
    Load raw JSONL file, clean sentences, and return a flat list of cleaned sentences.

    Args:
        input_path: Path to raw input JSONL.

    Returns:
        A list of cleaned, terminal-ending, tagged sentences.
    """
    cleaned_sentences: List[str] = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:100]}")
                continue

            # Identify any "chunk_xx" key
            chunk_key = next((k for k in data.keys() if k.startswith("chunk")), None)
            if not chunk_key:
                continue

            chunk = data.get(chunk_key, [])
            if not isinstance(chunk, list):
                chunk = [chunk]

            for sent in chunk:
                cleaned = clean_sentence(sent)

                if cleaned and has_terminal(cleaned):
                    # Append sentence tag
                    if not cleaned.endswith("<|sentence|>"):
                        cleaned = cleaned.rstrip() + "<|sentence|>"

                    cleaned = capitalize_first_letter(cleaned)
                    cleaned_sentences.append(cleaned)

    return cleaned_sentences


def build_v1(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Create dataset v1: pairs of sentences (2-by-2).

    Args:
        sentences: All cleaned sentences.

    Returns:
        List of {"text": "..."} entries.
    """
    output = []
    for i in range(0, len(sentences), 2):
        pair = sentences[i:i + 2]
        output.append({"text": " ".join(pair)})
    return output


def build_v2(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Create dataset v2: sentence_i + overlap(sentence_{i+1})

    Args:
        sentences: All cleaned sentences.

    Returns:
        List of {"text": "..."} entries.
    """
    output = []
    for i in range(len(sentences) - 1):
        overlap = get_overlap(sentences[i + 1])
        if overlap:
            output.append({"text": f"{sentences[i]} {overlap}"})
    return output


def build_v3(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Create dataset v3: same as v2 but stepping by 2.

    Args:
        sentences: All cleaned sentences.

    Returns:
        List of {"text": "..."} entries.
    """
    output = []
    for i in range(0, len(sentences) - 1, 2):
        overlap = get_overlap(sentences[i + 1])
        if overlap:
            output.append({"text": f"{sentences[i]} {overlap}"})
    return output


def save_jsonl(data: List[Dict[str, str]], path: str) -> None:
    """
    Save a list of dicts to a JSONL file.

    Args:
        data: List of {"text": "..."} entries.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Created {len(data)} entries → {path}")


def main(input_path: str) -> None:
    """
    Full pipeline:
        Raw JSONL → Clean sentences → Build v1/v2/v3 → Save output.

    Args:
        input_path: Path to input raw JSONL file.
    """
    logger.info("Loading & cleaning raw data.")
    sentences = load_and_clean_raw_data(input_path)

    logger.info(f"Collected {len(sentences)} cleaned sentences.")

    logger.info("Building datasets.")
    v1 = build_v1(sentences)
    v2 = build_v2(sentences)
    v3 = build_v3(sentences)

    save_jsonl(v1, "train_data_v1.jsonl")
    save_jsonl(v2, "train_data_v2.jsonl")
    save_jsonl(v3, "train_data_v3.jsonl")

    logger.info("Saved datasets: train_data_v1.jsonl, train_data_v2.jsonl, train_data_v3.jsonl")

if __name__ == "__main__":
    main("raw_data.jsonl")
