import json
from pathlib import Path
from utils.constants import SENTENCE_TAG, CONFUSABLE_MAP, ALLOWED_AZ_CHARS

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
                                break
                            
                    if valid_sentence:  # skip non-valid sentence
                        updated_sentences.append(s)

            training_data = {
                "input": " ".join(updated_sentences),
                "output": f"{SENTENCE_TAG} ".join(updated_sentences)
            }
            preprocessed_data.append(training_data)

        return preprocessed_data
    
if __name__ == "__main__":
    input_dir = Path(r'generation\results.jsonl')
    preprocessed = preprocessing(data_dir=input_dir)
    output_dir = Path(r'azcharboundary\data\data.jsonl')

    with open(output_dir, "w", encoding="utf-8") as f_out:
        for item in preprocessed:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")