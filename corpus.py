TERMINAL_SENTENCE_CHAR_LIST = frozenset(
    [
        # punctuation marks
        ".",  # period
        "!",  # exclamation mark
        "?",  # question mark
        ";",  # semicolon (often used in complex sentences)
        # quotations (straight and curly)
        '"',  # straight double quotes
        "\u201d",  # right double quotation mark (curly)
        "'",  # straight single quote
        "\u2019",  # right single quotation mark (curly)
        # other punctuation marks
        ":",  # colon (can end sentences in certain contexts)
        "...",  # ellipsis (can indicate a trailing off or incomplete thought)
    ]
)

with open("corpus.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
print(f'Line count: {len(lines)}')

# getting blocks
blocks: list[str] = []
block = []
for line in lines:
    if line == '\n':
        if block:
            blocks.append(block)
            block = []
    else:
        block.append(line)

print(f'Block count: {len(blocks)}')


chunked_blocks: dict = {}
chunk = ""
for block in blocks:
    pass