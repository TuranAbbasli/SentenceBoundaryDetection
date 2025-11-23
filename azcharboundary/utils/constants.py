"""
Constants used throughout the charboundary library.
"""

AZ_ALPHABET = [
    "a", "b", "c", "ç", "d", "e", "ə", "f", "g", "ğ", "h", "x",
    "ı", "i", "j", "k", "q", "l", "m", "n", "o", "ö", "p", "r",
    "s", "ş", "t", "u", "ü", "v", "y", "z"
]

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

# whitespace
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
        "\n",  # newline character (common in text files)
        "\r",  # carriage return (used in some text formats)
    ]
)

# Primary terminators - more likely to end sentences
PRIMARY_TERMINATORS = frozenset([".", "!", "?", "\n", "\r"])

# Secondary terminators - less likely to end sentences on their own
SECONDARY_TERMINATORS = frozenset(['"', "\u201d", "'", "\u2019", ";", ":"])

# Opening quotation marks
OPENING_QUOTES = frozenset(['"', "\u201c", "'", "\u2018"])

# Closing quotation marks
CLOSING_QUOTES = frozenset(['"', "\u201d", "'", "\u2019"])

# Annotation tags
SENTENCE_TAG = "<|sentence|>"

CONFUSABLE_MAP = {
    # Cyrillic to Latin (Uppercase)
    'А': 'A',  'В': 'B',  'Е': 'E',  'К': 'K',  'М': 'M',
    'Н': 'H',  'О': 'O',  'Р': 'P',  'С': 'C',  'Т': 'T',
    'У': 'U',  'Х': 'X',
            
    # Cyrillic to Latin (Lowercase)
    'а': 'a',  'е': 'e',  'о': 'o',  'р': 'p',  'с': 'c',
    'у': 'u',  'х': 'x',  'ѕ': 's',  'і': 'i',  'ј': 'j',
            
    # Greek to Latin
    'Α': 'A',  'Β': 'B',  'Ε': 'E',  'Ζ': 'Z',  'Η': 'H',
    'Ι': 'I',  'Κ': 'K',  'Μ': 'M',  'Ν': 'N',  'Ο': 'O',
    'Ρ': 'P',  'Τ': 'T',  'Υ': 'Y',  'Χ': 'X',
            
    # Specific confusables
    'ï': 'i', 
    'ı̇': 'i',
            
    # Common OCR mistakes for Azerbaijani special chars
    'ә': 'ə',  'Ә': 'Ə',  'ǝ': 'ə',  'ɘ': 'ə'
}

ALLOWED_AZ_CHARS = {
    # Azerbaijani Letters (Uppercase)
    'A', 'B', 'C', 'Ç', 'D', 'E', 'Ə', 'F', 'G', 'Ğ','H', 'X', 'İ', 'I','J', 'K',
    'Q', 'L', 'M', 'N', 'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U', 'Ü', 'V', 'Y', 'Z',

    # Azerbaijani Letters (Lowercase)
    'a', 'b', 'c', 'ç', 'd', 'e', 'ə', 'f', 'g', 'ğ', 'h', 'x', 'ı', 'i', 'j', 'k',
    'q', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z'

}

# Default list of common abbreviations that end with a period but don't end sentences
DEFAULT_ABBREVIATIONS = [
    "(A).", "-Ş.", "0.2Sta.", "0Bis.", "130-2fər.", "130f.", "16f.", "18x22sm.", "1L.", "1a.",
    "2-Bis.", "28-bis.", "3-Bis.", "3.2.B.", "3.a.", "3/4-in.", "3Bis.", "3\\T.", "3⁄8-in.",
    "42KB.", "42KC.", "6Bis.", "6Ter.", "6sm.", "6ter.", "7S.R.L.", "9/T.K.", "95sm.", "A-l.",
    "A.", "A.A.", "A.A.S.A.", "A.B.", "A.Bölməsi.", "A.C.", "A.E.", "A.F.", "A.G.", "A.H.",
    "A.K.", "A.L.P.E.E.", "A.M.", "A.O.", "A.Q.", "A.R.", "A.S.", "A.S.A.", "A.T.", "A.X.",
    "A.Y.", "A.Z.", "A.l.", "A.İ.", "A.Ş.", "A.Ə.", "ARCin.", "ASC.", "AZ.", "Acc.", "Ak.",
    "Akad.", "Art.", "Arx.", "As.", "Assoc.", "Az.", "Azərb.", "B.", "B.2.", "B.2.1.", "B.4.",
    "B.5.", "B.A.", "B.C.", "B.I.", "B.III.", "B.IV.", "B.M.", "B.V.", "B.Y.", "B.l.", "B.Ş.",
    "B.Ə.", "B2.", "Bax.", "Bdəy.", "Bhd.", "Bi.", "Bi.Vi.", "Bis.", "Bldg.", "BÖLMƏ.", "C.",
    "C.2.", "C.5.", "C.C.", "C.F.", "C.F.R.", "C.M.", "C.Q.", "C.l.", "C.İ.F.", "C.Ə.", "CO.",
    "COI/T.20/Doc.", "COI/T.20/Doc.no.", "COƏ/T.20/Doc.", "Chem.Soc.", "Co.", "Coemeandothersv.",
    "Corc.", "Crt.", "D.", "D.C.", "D.F.", "D.H.", "D.M.", "D.P.", "D.Q.", "D.V.", "D.c.", "D.l.",
    "D2.", "D2.1.", "D2.2.", "D3.2.", "DTK.", "Desf.", "Dm/avad.", "Dm/yum.", "Dma.", "Dmma.",
    "Dmöv.avad.", "Dmöv.maş.avad.", "Dmöv.yum.", "Doc.", "Doc.no.", "Dr.", "Dəf.", "E.",
    "E.4.2.", "E.A.", "E.B.", "E.F.", "E.H.", "E.H.Q.", "E.L.", "E.M.", "E.N.", "E.S.", "E.T.",
    "E.T.H.M.", "E.U.", "E.Z.", "E.d.", "E.İ.", "E.Ə.", "ETSNo.", "Edax.o.", "Ef.b.", "Eist.",
    "ElTeDe.", "Exüs.s.", "F.", "F.16.", "F.A.", "F.A.A.", "F.C.", "F.F.", "F.H.", "F.J.", "F.M.",
    "F.S.", "F.X.", "F.Y.", "F.Z.", "F.İ.", "F.Ə.", "F2.1.", "Fut/san.", "FƏSİL.", "G.", "G.E.",
    "G.H.H.", "G.R.", "G.S.", "G.S.U.", "GNKUR.", "Gar.", "H.", "H.A.", "H.B.", "H.C.", "H.F.",
    "H.H.", "H.P.", "H.S.", "H.X.", "H.Z.", "H.İ.", "H.Ə.", "HA.", "Hab.", "Hb/tel.", "Hbey.tel.",
    "Hbt.", "Hdx.", "He.", "Hes.", "Hint.", "Host.", "Htel.", "I-I.", "I.", "I.I.", "II.", "III-I.",
    "III.", "IV.", "IX.", "Ii.", "Il.", "Iv.", "Ix.", "J.", "J.Amer.", "J.E.", "J.J.P.", "J.L.",
    "J.M.", "J.V.", "K.", "K.B.", "K.M.", "K.Ş.", "KM.", "KMQ1.", "KMQ15.", "KMQ2.", "Kamort.",
    "Khes.ist.", "Ko.", "Kt.i.", "Kub.", "Kva.", "L.", "L.-Ş.", "L.C.B.", "L.M.", "L.d.", "L.l.",
    "L.İ.", "LI.", "LIII.", "LIV.", "LTD.", "LV.", "LVI.", "LX.", "LXI.", "LXII.", "LXIII.", "LXIV.",
    "LXV.", "LXVI.", "LXVII.", "Ltd.", "M.", "M.A.", "M.B.", "M.C.", "M.C.M.", "M.F.", "M.H.",
    "M.K.", "M.M.", "M.P.", "M.S.", "M.T.", "M.V.", "M.X.", "M.Y.", "M.Z.", "M.İ.", "M.Ş.", "M.Ə.",
    "M/San.", "M7.", "MD.", "ML7.h.1.", "Mad.", "Mbs.", "Min.", "Mln.", "Məs.", "N.", "N.A.",
    "N.B.", "N.C.", "N.F.", "N.H.", "N.K.", "N.K.M.", "N.R.", "N.U.", "N.V.", "N.İ.", "N1g.",
    "N1gə.", "NYX.", "Na/xid.", "Navad.", "Nax.", "Nb/tel.", "Nbey.tel.", "Nbt.", "Nct.", "Ndt.",
    "Ndəf./təsər.", "Ne.", "Nez.", "Nezam.", "Nint.", "Nma.", "Nmaş./avad.", "Nmaş./avad..",
    "Nmaş.avad.", "Nmm.", "Nmət.", "Nnəq.", "No.", "Ntel.", "Nyan.", "Nys.", "Nyum.", "Nöt.",
    "Nİ–bax.", "Nə.", "Nə.üst.", "Nərz.", "O.", "O.N.", "O.X.", "O.Z.", "Oil.Soc.", "P-2.", "P-2f.",
    "P-3f.", "P-7f.", "P.", "P.D.", "P.E.E.", "P.Henn.", "P.J.", "P.M.", "P.O.", "PMO_1.", "PMO_2_1.",
    "PMO_4.", "PMO_4_2.", "PMO_5.", "PMO_7.", "PROf.", "PTİ.", "PTİ.LTD.", "Phes.q.y.", "Prof.",
    "Q.", "Q.A.", "Q.D.", "Qa.", "Qaqyh.", "Qavad.", "Qeyd.", "Ql/kab.", "Qma.", "Qmm.", "Qmə.",
    "Qr.", "Qtm.", "Quater.", "Quin.", "Qxg.", "Qy.", "Qyd.", "Qyl.", "Qyum.", "Qəm.", "R.", "R.A.",
    "R.B.", "R.C.", "R.E.", "R.H.", "R.K.", "R.M.", "R.Q.", "R.R.", "R.S.", "R.T.", "R.V.", "R.Z.",
    "R.İ.", "R.Ş.", "R.Ə.", "REV.", "Ref.", "Resp.", "Respub.", "Rev.", "S.", "S.A.", "S.A.A.",
    "S.A.L.", "S.C.", "S.H.", "S.J.", "S.M.", "S.N.", "S.R.", "S.R.L.", "S.S.", "S.T.", "S.c.",
    "S.Ç.", "S.İ.", "SHg.", "SHg.g.d.", "ST.", "Sas.", "Savad.", "Sb/tel.", "Sbey.tel.", "Sbt.",
    "Sdn.", "Set.", "Sex.", "Sez.", "Sezam.", "Sg.", "Sgün.", "Sint.", "Sl/kab.", "Sm.t.", "Sma.",
    "Smm.", "Smə.", "Sn.", "Sov.", "Sr.", "St.", "Sta.", "Stel.", "Stm.", "Suşaq.", "Sy.", "Syl.",
    "SÖg.", "Sç.", "Sçar.", "Süt.", "Sşag.", "Səh.", "Sət.", "T.", "T.A.", "T.F.", "T.K.", "T.M.",
    "T.N.", "T.P.", "T.R.", "T.Y.", "T.İ.", "T.Ə.", "TIAXg.", "TSHg.", "TSÖg.", "TZTXg.", "Ter.",
    "Thirumet.", "Tmiq.", "Tq.m.", "TİAXg.", "U.", "U.S.", "U.S.C.", "U.S.S.", "Usın.", "V-I.", "V.",
    "V.A.", "V.B.", "V.D.", "V.H.", "V.J.", "V.Q.", "V.R.", "V.S.", "V.V.", "V.f.", "V.Ç.", "V.İ.",
    "V.Ə.", "V/el.", "VI-II.", "VI.", "VII.", "VIII.", "Verein.", "Verh.", "Vi.", "Vl.", "Vll.", "X-I.",
    "X-II.", "X.", "X.A.", "X.B.", "X.H.", "X.M.", "X.R.", "X.S.", "X.İ.", "XI-II.", "XI.", "XII-I.",
    "XII.", "XIII.", "XIV.", "XIX.", "XL.", "XLII.", "XLIII.", "XLIX.", "XV.", "XVI.", "XVII.",
    "XVIII.", "XX.", "XXI.", "XXII.", "XXIII.", "XXIV.", "XXV.", "XXVI.", "XXVII.", "XXX.",
    "XXXI.", "XXXII.", "XXXIV.", "Xfakt.", "Xs.", "Y.", "Y.C.", "Y.K.", "Y.M.", "Y.Q.", "Y.T.",
    "Y.V.", "Y.İ.", "Yn.", "Z.", "Z.Q.", "Z.R.", "Z.T.", "Z.Y.", "Z.Ə.", "ZA.", "a.", "a.c.", "a.f.",
    "a.g.", "a.q.", "a.s.", "ad.", "akad.", "akt.", "alk.", "atas.", "atm.", "avad.", "b.", "b.3.",
    "b.k.", "b/tel.", "bax.", "bis.", "c.", "c.p.", "cf.", "cüm.", "d.", "d.c.", "d.l.", "d.o.o.",
    "d/dəq.", "diam.", "dok.", "dr.", "döv/dəq.", "düym.", "dəq.", "dər.", "e.", "e.h.q.", "e.m.f.",
    "e.ə.", "f.", "fad.", "fam.", "fiq.", "fər.", "g.", "h.", "h.ş.", "hes.", "his.", "hə.", "i.",
    "i.a.", "ii.", "iii.", "il.", "in.", "ist.", "iv.", "j.", "k.", "k.g.", "kVt.", "kab.", "kd.",
    "km.", "kq.", "kq/kv.sm.", "kqs/kv.sm.", "kub.", "kub.m.", "kv.", "kv.m.", "kv.sm.", "kvt.",
    "kvt.s.", "küç.", "l.", "lk.", "m-2.", "m.", "m.man.", "m.y.", "m/san.", "m3/san.", "mad.",
    "maks.", "man.", "mayor.", "metr.", "min.", "mkr.", "ml.", "mld.", "mln.", "mlrd.", "mm.",
    "mm/dəq.", "mq.l.", "məh.", "mən.", "məs.", "n.", "no.", "o.", "o.k.d.", "o.k.q.", "o.kv.",
    "op.", "p.", "p.e.n.", "paq.", "pdf.", "poq.", "pp.", "ppm.", "pr.", "prof.", "q.", "q.r.",
    "qar.", "qiy.", "qr.", "qsah.", "qəs.", "r.", "r.m.s.", "rad.", "resp.", "rub.", "rüb.", "s.",
    "s.a.", "s.a.a.", "s.a.a.a.", "s.c.", "s.c./d.c.", "s.r.o.", "sair.", "san.", "sex.", "sm.",
    "sp.", "ss.", "st.", "süt.", "səh.", "sər.", "sət.", "t.", "t.g.", "t/h.", "tel.", "tn.kq.ədəd.",
    "tor.", "trln.", "təqr.", "təxm.", "təş.", "v.", "v.s.", "v.ü.", "var.", "vasit.", "vs.", "və.",
    "və.s.", "və.s...", "zərbə/dəq.", "Ç.", "Ö.", "ÖEŞtex.itki.", "Ü.", "Ü.H.", "Üz.", "ç.", "Ğ.",
    "İ.", "İ.B.", "İ.D.", "İ.F.", "İ.H.", "İ.M.", "İ.N.", "İ.R.", "İAXg.", "İNK.", "İnc.", "İnk.",
    "İnternat.", "İnv.", "Ş.", "Ş.H.", "Ş.K.", "Ş.M.", "Ş.Ş.", "Ş.Ə.", "Şək.", "Şəkil.", "ş.", "şəh.",
    "şək.", "Ə.", "Ə.A.", "Ə.H.", "Ə.M.", "Ə.Q.", "Ə.R.", "Ə.T.", "Ə.Y.", "Ə.İ.", "Ə.Ş.", "Ə.Ə.",
    "Ə2.", "ƏDV.", "Əamort.", "Əeh.ct.", "Əinf.", "Əməl.", "Əor.", "Əs.", "ə.", "əlavə.", "əms.",
    "əməl.", "əs.", "əsas.", "В.1.", "С.", "№tel."
]


# Enumeration patterns - used to detect list items
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

# Conjunction patterns that often appear in the last item of a list
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
