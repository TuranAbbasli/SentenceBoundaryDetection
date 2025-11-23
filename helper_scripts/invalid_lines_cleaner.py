import json

input_path = "data/output_from_types_generation.jsonl"
output_path = "data/output_from_types_cleaned.jsonl"

num_valid = 0
num_invalid = 0
num_bad_encoding = 0

with open(input_path, "rb") as infile, open(output_path, "w", encoding="utf-8") as outfile:

    for i, raw_line in enumerate(infile, start=1):
        # Step 1: Try UTF-8 decode
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            num_bad_encoding += 1
            print(f"Bad UTF-8 on line {i}, dropping.")
            continue

        stripped = line.strip()
        if not stripped:
            num_invalid += 1
            continue

        # Step 2: Try parse JSON
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            num_invalid += 1
            print(f"Invalid JSON on line {i}: {stripped[:120]}...")
            continue

        # Step 3: line is valid; keep it
        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
        num_valid += 1

print("\n--- SUMMARY ---")
print(f"Valid lines written:       {num_valid}")
print(f"Invalid JSON dropped:      {num_invalid}")
print(f"Bad encoding lines dropped:{num_bad_encoding}")