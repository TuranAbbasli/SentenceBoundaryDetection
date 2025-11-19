
#Tags
SENTENCE_TAG = "<|sentence|>"


# Punctuation list, including unicode
PUNCTUATION_CHAR_LIST = frozenset(
    [
        ")",
        "]",
        "}",
        "\u0f3b",
        "\u0f3d",
        "\u169c",  # Unicode brackets and close quotes
        "\u2046",
        "\u207e",
        "\u208e",
        "\u2309",
        "\u230b",
        "\u3009",
        "\u2769",
        "\u276b",
        "\u276d",
        "\u276f",
        "\u2771",
        "\u2773",
        "\u2775",
        "\u27c6",
        "\u27e7",
        "\u27e9",
        "\u27eb",
        "\u27ed",
        "\u27ef",
        "\u2984",
        "\u2986",
        "\u2988",
        "\u298a",
        "\u298c",
        "\u298e",
        "\u2990",
        "\u2992",
        "\u2994",
        "\u2996",
        "\u2998",
        "\u29d9",
        "\u29db",
        "\u29fd",
        "\u2e23",
        "\u2e25",
        "\u2e27",
        "\u2e29",
        "\u2e56",
        "\u2e58",
        "\u2e5a",
        "\u2e5c",
        "\u3009",
        "\u300b",
        "\u300d",
        "\u300f",
        "\u3011",
        "\u3015",
        "\u3017",
        "\u3019",
        "\u301b",
        "\u301e",
        "\u301f",
        "\ufd3e",
        "\ufe18",
        "\ufe36",
        "\ufe38",
        "\ufe3a",
        "\ufe3c",
        "\ufe3e",
        "\ufe40",
        "\ufe42",
        "\ufe44",
        "\ufe48",
        "\ufe5a",
        "\ufe5c",
        "\ufe5e",
        "\uff09",
        "\uff3d",
        "\uff5d",
        "\uff60",
        "\uff63",
        ".",
        ")",
        "\u00bb",
        "\u2019",
        "\u201d",
        "\u203a",
        "\u2e03",
        "\u2e05",
        "\u2e0a",
        "\u2e0d",
        "\u2e1d",
        "\u2e21",
        "\u201c",
        "\u201d",
        "_",
        "\u203f",
        "\u2040",
        "\u2054",
        "\ufe33",
        "\ufe34",
        "\ufe4d",
        "\ufe4e",
        "\ufe4f",
        "\uff3f",
        ":",
        ";",
        ",",
        "&",
        "-",
        "\u05be",
        "\u05bf",
        "\u1400",
        "\u1806",
        "\u2010",
        "\u2011",
        "\u2012",
        "\u2013",
        "\u2014",
        "\u2015",
        "\u2e17",
        "\u2e1a",
        "\u2e3a",
        "\u2e3b",
        "\u2e40",
        "\u2e5d",
        "\u301c",
        "\u3030",
        "\u30a0",
        "\ufe31",
        "\ufe32",
        "\ufe58",
        "\ufe63",
        "\uff0d",
        "\U00010e2d",
        "\u002d",
        "\u2013",
        "\u2014",
    ]
)

WS_CHAR_LIST = frozenset(
    [
        " ",
        "\xa0",
        "\u1680",
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",
        "\u205f",
        "\u3000",
        "\t",
        "\u2028",
        "\r",
        "\n",
    ]
)


# list of characters that can possibly end a sentence
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

# list of characters that can indicate the end of a paragraph
TERMINAL_PARAGRAPH_CHAR_LIST = frozenset(
    [
        # characters that can end a paragraph
        "\n",  # newline character (common in text files)
        "\r",  # carriage return (used in some text formats)
        # terminal sentence characters can also indicate end of paragraph
        # if they appear at the end of a line
    ]
)
DEFAULT_ABBREVIATIONS = [
# --- Roman numerals ---
"I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.",
"XIII.", "XIV.", "XV.", "XVI.", "XVII.", "XVIII.", "XIX.", "XX.", "XXI.", "XXII.",
"XXIII.", "XXIV.", "XXV.", "XXVI.", "XXVII.", "XXIX.", "XXX.", "XXXI.", "XXXII.",
"XXXIII.", "XXXIV.", "XXXV.", "XXXVI.",

# --- Unit & measurement abbreviations ---
"km.", "kq.", "sm.", "mm.", "mq.", "mln.", "mlrd.", "man.", "qəp.", "kv.", "kV.",
"kVt.", "kub.", "qr.", "qrup.", "tn.", "uçot.",

# --- Time, numbering ---
"s.", "səh.", "dəq.", "No.",

# --- Titles, professions, roles ---
"prof.", "Prof.", "Dr.", "dr.", "Sov.", "İmza.", "İdx.", "Tel.", "tel.",

# --- Organizations / institutions ---
"ASC.", "QSC.", "PLC.", "CO.", "Co.", "PR.", "GOV.", "gov.", "AZ.", "IIQR.",
"QOŞMA.", "XHK.", "XAKS.", "İNF.", "PM.", "TRDM.", "INK.", "İNK.", "KM.", "MY.",
"II.", # occasionally institutional section

# --- Common Azerbaijani print shorthand ---
"şək.", "süt.", "hes.", "resp.", "şəh.", "akt.", "məs.", "mad.", "sətr.",
"şüşə.", "otaq.", "otağ.", "şək.", "bank.", "plan.", "zona.", "zərf.", "əms.",
"faiz.", "üzrə.", "üzvü.",

# --- Foreign / technical abbreviations ---
"www.", "org.", "Org.", "etc.", "edu.", "Ltd.", "off.", "Rec.", "Doc.",

# --- Russian abbreviations found in corpus ---
"г.", "Г.", "гор.", "тыс.", "млн.", "руб.", "СССР.", "АЭС.",

# --- Mixed-case and special uppercase abbreviations ---
"AZ.", "KM.", "PR.", "QME.", "MY.", "FH.", "WİF.", "PLC.", "ATM.", "TYK.", "KM.",
"Borca.", "BORCU.",

# --- Uppercase short abbreviations (heuristic-selected) ---
"A.", "M.", "S.", "F.", "T.", "R.", "N.", "H.", "B.", "E.", "Ə.", "C.", "X.",
"Ş.", "Z.", "W.", "P.", "G.", "L.", "Ü.", "Ç.", "Q.", "İ.", "U.", "J.", "O.",

# --- Multi-letter uppercase abbreviations found ---
"GOV.", "CO.", "AZ.", "ASC.", "QSC.", "PLC.", "PR.", "QME.", "WİF.", "Süt.",
"Sov.", "IIQR.", "II.", "MY.", "KM.", "PM.", "TRDM.", "INK.", "İNK."]




# Primary terminators - more likely to end sentences
PRIMARY_TERMINATORS = frozenset([".", "!", "?"])

# Secondary terminators - less likely to end sentences on their own
SECONDARY_TERMINATORS = frozenset(['"', "\u201d", "'", "\u2019", ";", ":"])

# Opening quotation marks
OPENING_QUOTES = frozenset(['"', "\u201c", "'", "\u2018"])

# Closing quotation marks
CLOSING_QUOTES = frozenset(['"', "\u201d", "'", "\u2019"])


LIST_MARKERS = [
    # Numbered markers (parentheses)
    "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "(10)",

    # Roman numerals (parentheses)
    "(i)", "(ii)", "(iii)", "(iv)", "(v)", "(vi)", "(vii)", "(viii)", "(ix)", "(x)",

    # Letter markers (parentheses)
    "(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)",

    # Numbered markers (period)
    "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",

    # Letter markers (period)
    "a.", "b.", "c.", "ç.", "d.", "e.", "ə.", "f.", "g.", "h.", "i.", "j.",

    # Roman numerals (period)
    "i.", "ii.", "iii.", "iv.", "v.", "vi.", "vii.", "viii.", "ix.", "x.",

    # Numbered markers (closing parenthesis)
    "1)", "2)", "3)", "4)", "5)", "6)", "7)", "8)", "9)", "10)",

    # Letter markers (closing parenthesis)
    "a)", "b)", "c)", "ç)", "d)", "e)", "ə)", "f)", "g)", "h)", "i)", "j)",

    # Bullet symbols
    "•", "·", "○", "●", "■", "□", "▪", "▫",
]



LIST_INTROS = [
    # Direct translations
    "aşağıdakı:",                     # following:
    "aşağıdakı kimi:",                # as follows:
    "aşağıdakıların siyahısı:",       # the following items:
    "aşağıdakılar:",                  # the following:
    "aşağıda göstərilənlər:",         # items listed below:
    "aşağıda sadalananlar:",          # listed below:
    "aşağıda qeyd edilənlər:",        # items below:
    "daxildir:",                      # include:
    "daxil olmaqla:",                 # including:
    "o cümlədən:",                    # including / inter alia:
    "məsələn:",                       # such as:
    "yəni:",                           # namely:

    # Legal/administrative Azerbaijani formal patterns
    "aşağıdakılar müəyyən edilir:",   # the following are established:
    "aşağıdakılar tətbiq olunur:",    # the following apply:
    "aşağıdakılar nəzərdə tutulur:",  # the following are envisaged:
    "aşağıdakılar tənzimləyir:",      # the following regulate:
    "aşağıdakılar müəyyən edilir ki:",# it is determined as follows:
    "bu Qaydalara əsasən aşağıdakılar:", # according to these rules, the following:
    "bəyan olunur ki:",               # it is declared that:
    "tələb olunur ki:",               # it is required that:
    "müəyyən edilir ki:",             # it is determined that:
    "qeyd olunur ki:",                # it is noted that:

    # Contract / procurement / SBD-specific phrasing
    "bu sənədə əsasən aşağıdakılar:", # pursuant to this document:
    "bu bölmədə aşağıdakılar göstərilir:", # in this section:
    "aşağıdakı şərtlər tətbiq olunur:",    # the following conditions apply:
    "aşağıdakı tələblər yerinə yetirilməlidir:", # the following requirements must be met:
    "aşağıdakı sənədlər təqdim edilməlidir:",    # the following documents must be submitted:
    "tərəflər aşağıdakılar barədə razıdır:",     # parties agree on the following:
    "işlər aşağıdakılardan ibarətdir:",         # works consist of the following:

    # Extended variants typical in laws/regulations
    "aşağıdakı hallarda:",               # in the following cases:
    "aşağıdakı əsaslarla:",              # on the following grounds:
    "aşağıdakılar çərçivəsində:",        # within the following framework:
    "aşağıdakı məlumatlar:",             # the following information:
    "aşağıdakı müddəalar:",              # the following provisions:
    "aşağıdakı kateqoriyalar:",          # the following categories:
    "aşağıdakı öhdəliklər:",             # the following obligations:
    "aşağıdakı hüquqlar:",               # the following rights:
]


LIST_CONJUNCTIONS = [
    # Basic conjunctions
    " və ",              # and
    " və ya ",           # or
    " ya da ",           # or
    " yaxud ",           # or
    " eləcə də ",        # as well as
    " həmçinin ",        # also / as well

    # Less formal / conversational variants
    " bir də ",          # also / and (colloquial)
    " üstəlik ",         # moreover / in addition
]
