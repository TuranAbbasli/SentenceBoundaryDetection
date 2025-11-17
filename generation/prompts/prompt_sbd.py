ASSISTANT_TEMPLATE="""
You are an expert in Azerbaijani legal-domain sentence boundary detection. 
Split the given text chunk into sentences while preserving the original text EXACTLY.
"""

PROMPT_TEMPLATE="""
=== CRITICAL REQUIREMENTS ===

**OUTPUT FORMAT**
- JSON array of strings ONLY: ["sentence 1", "sentence 2", "sentence 3"]
- NO explanations, markdown, or commentary
- NO quotation marks or brackets added to sentence content
- NO translation, modification, or character changes
- Preserve text 100% exactly — same words, spacing, case, punctuation, line breaks

**PROCESSING METHOD**
1. Analyze the full context before splitting
2. Apply legal and linguistic boundary rules consistently across the chunk

=== SPLITTING RULES ===

**TERMINATORS**
- Primary terminators: ".", "!", "?"
- Secondary (context-dependent): ";", "\"", "\u201d", "'", "\u2019", ":", "...",
- Keep complex legal provisions together if they form single legal statement
- Keep numbering (Maddə 5, Bölmə III) attached to content

**ABBREVIATIONS - DO NOT SPLIT**
m., art., b., p., par., №, səh., il., prof., dr., ə.e.d., AzSSR., AMEA.,
AG., AK., mül., cin., q., f., v.s., və s., və b., initials (A.M., H.Ə.) and similar

**Structure & Formatting**
- Underscores (____) are fillable gaps → keep inside same sentence
- Preserve all whitespace (including double or non-breaking)
- Do not split inside parentheses or quotation marks
- If a quote ends after a period (e.g., ."), treat it as one boundary

**LISTS & ENUMERATIONS**
- First analyze: are list items phrases or complete sentences?
- Phrase lists (short items): a) x; b) y; c) z → ONE sentence including intro
- Sentence lists (grammatically complete items) → SEPARATE sentence per item
- Multi-line items: split only if each part forms valid, complete sentence structure
- When ambiguous: if items can stand alone meaningfully → split; otherwise → keep together

**LEGAL STRUCTURES**
- Keep hierarchical numbering ("1.1.", "45.2.3.") intact
- Definitions ("X termini ... deməkdir") → typically one sentence
- Keep citations with their sentence
- Keep parenthetical content with host sentence
- Institution names and dates (15 mart 2024-cü il, 21.05.2023) → no split
- Long legal provisions → keep as ONE if single legal statement

**. Priority**
1. Preserve text exactly  
2. Maintain semantic and legal completeness  
3. When unsure, keep together

=== INPUT ===
{input_chunk}

=== OUTPUT ===
Return JSON array only — one complete, unmodified sentence per element.
"""