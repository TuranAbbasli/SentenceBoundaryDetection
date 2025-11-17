LEGAL_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "indexes": {
            "type": "array",
            "description": (
                "List of indexes of the input records that meet the filtering criteria."
            ),
            "items": {
                "type": "integer",
                "description": (
                    "Index of a single input record that meets the filtering criteria."
                )
            },
            "uniqueItems": True   # each index should appear only once
        }
    },
    "required": ["indexes"],
    "additionalProperties": False
}

SPELLING_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "The fully processed Azerbaijani version of the input query. "
                "It must be grammatically and orthographically correct, all lowercase, "
                "and preserve search logic, punctuation, and legal operators exactly."
            )
        },
        "was_translated": {
            "type": "boolean",
            "description": (
                "Indicates whether the original query contained any non-Azerbaijani elements "
                "that were translated into Azerbaijani. "
                "True if translation occurred; false if only corrections were applied."
            )
        }
    },
    "required": ["query", "was_translated"],
    "additionalProperties": False
}

SBD_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "description": (
                "List of sentences extracted from the input text. "
                "Each element must be a string representing one sentence, "
                "preserved exactly as in the original text without rewriting, "
                "except trimming leading/trailing whitespace."
            ),
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["sentences"],
    "additionalProperties": False
}

ABBRV_SCHEMA = {
    "type": "object",
    "properties": {
        "abbreviations": {
            "type": "array",
            "description": (
                "List of abbreviations extracted from the input text. "
                "Each element must be a string representing one abbreviation, "
                "preserved exactly as it appears in the original text without modification. "
                "Each abbreviation should appear only once. "
                "Return an empty array if no abbreviations are found."
            ),
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["abbreviations"],
    "additionalProperties": False
}
