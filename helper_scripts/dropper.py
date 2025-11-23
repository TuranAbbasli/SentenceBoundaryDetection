input_path = "data/train_data_v2.jsonl"
output_path = "train_data_v2_no_t.jsonl"

DROP = 278_500

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for i, line in enumerate(infile):
        if i < DROP:
            continue   # skip
        outfile.write(line)

print("Done. Wrote all entries after", DROP)