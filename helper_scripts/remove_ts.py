import json

input_path = "data/output_from_types_cleaned.jsonl"
output_path = "data/train_data_labeled_types.jsonl"

def remove_tabs(obj):
    if isinstance(obj, str):
        return obj.replace("\t", "").replace("\\t", "")
    elif isinstance(obj, list):
        return [remove_tabs(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: remove_tabs(v) for k, v in obj.items()}
    return obj

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for i, line in enumerate(infile, start=1):
        line_strip = line.strip()

        # skip empty or whitespace-only lines
        if not line_strip:
            print(f"Skipping empty line {i}")
            continue

        try:
            data = json.loads(line_strip)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON on line {i}: {line_strip[:80]}...")
            continue

        cleaned = remove_tabs(data)
        outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
