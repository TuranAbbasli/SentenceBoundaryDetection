import json
from pathlib import Path
from azcharboundary.utils.constants import SENTENCE_TAG, CONFUSABLE_MAP, ALLOWED_AZ_CHARS

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

            chunk = data.get("input")

            if not chunk:
                print(f'Empty value in generation data on line: {i}.')
                continue
            
            chunk = ''.join(CONFUSABLE_MAP.get(char, char) for char in chunk) 

            valid_chunk = True
            for char in chunk:
                if char.isalpha():
                    if char not in ALLOWED_AZ_CHARS:
                        valid_chunk = False
                        break
                            
            if valid_chunk:  # skip non-valid sentence
                data["input"] = chunk
                preprocessed_data.append(data)

        return preprocessed_data
    
if __name__ == "__main__":
    input_dir = Path('azcharboundary/data/train_data_labeled_types_abbr_cleaned.jsonl')
    preprocessed = preprocessing(data_dir=input_dir)
    output_dir = Path('azcharboundary/data/train_data_labeled_types_abbr_cleaned_fixed.jsonl')

    with open(output_dir, "w", encoding="utf-8") as f_out:
        for item in preprocessed:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")