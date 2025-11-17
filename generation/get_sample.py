import json

SAMPLE_DIR = 'generation/sample.jsonl'

sample: list[dict] = []

max_lines = 10_000

with open("generation/sentences.jsonl", "r", encoding="utf-8", errors="replace") as f:
    for idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue  # skip blank lines

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON on line {idx}: {e}")
            continue

        sample.append(data)

        if len(sample) == max_lines:
            break

print(f"Sample with length {len(sample)} was created. Starting writing!")

with open(SAMPLE_DIR, "w", encoding="utf-8") as out:
    for item in sample:
        out.write(json.dumps(item, ensure_ascii=False) + "\n")