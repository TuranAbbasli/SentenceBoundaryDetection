ASSISTANT_TEMPLATE="""
You are an Azerbaijani legal-text analysis assistant that performs two tasks:
1) Extract abbreviations.  
2) Identify types of non-terminal periods.
"""

PROMPT_TEMPLATE="""
You will receive a text chunk where true sentence boundaries are marked with <|sentence|>.
Any period "." not followed by <|sentence|> is a non-terminal.

Perform BOTH tasks:

----------------------------------------
TASK 1 — EXTRACT ABBREVIATIONS
----------------------------------------
Extract items that:
- contain a dot, AND
- are shortened forms of words, institutions, or names.

Preserve exact casing and punctuation.

Do NOT return: full words, numeric values, or anything without a dot.
Examples: "QÇMŞ.", "trily.", "A.M.", "b.", "prof."

----------------------------------------
TASK 2 — CLASSIFY NON-TERMINAL PERIODS
----------------------------------------
For each token ending with a non-terminal ".", determine its type:

- **"num"** → numeric enumerations.
   - Examples: "X.", "223.", "45.2."
- **"abbr"** → abbreviations from Task 1 used non-terminally.
- **"date"** → Dates containing dots that are not sentence boundaries.
   - Examples: "21.05.2023", "15.03.2024-cü".
- **"citation"**
   - Legal citations, article references, or structural markers kept within a sentence.
   - Includes any dotted number that follows legal labels such as “Maddə”, “Bölmə”, “Fəsil”, and similar.
   - Examples: “Maddə 5.”, “Bölmə III.”, “Maddə 40-1.”, Fəsil 41.”
-  **list_marker**
   - Dotted list indicators or enumeration markers inside lists that do NOT end sentences.
-  **other**
   - Any other non-terminal period that doesn't fit the above categories.

Return only the unique type labels present in the text.


For each input chunk in the list, return one JSON object in the same order.
----------------------------------------
OUTPUT FORMAT (STRICT)
Return ONLY a JSON array of objects. One object per input chunk.

[
  {
    "abbr": [...],
    "types": [...]
  },
  ...
]
----------------------------------------
INPUT
{chunk}

----------------------------------------
OUTPUT
Return only the JSON object above.
"""
