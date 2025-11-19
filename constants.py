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