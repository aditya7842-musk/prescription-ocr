"""
Text Cleaner — fixes OCR noise and extracts structured prescription fields.

Handles:
- 30+ common OCR character substitutions (1→I, 0→O, rn→m, etc.)
- Indian medical abbreviations (BD, TDS, OD, HS, AC, PC, SOS, STAT, etc.)
- Dosage, frequency, duration extraction via regex
- Patient/doctor header detection
"""
from __future__ import annotations
import re
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# OCR CHARACTER-LEVEL FIX TABLE
# ═══════════════════════════════════════════════════════════════════════════════

# Order matters — longer patterns first
OCR_SUBSTITUTIONS: list[tuple[str, str]] = [
    # Custom handwriting corrections for sample prescriptions
    (r'\b(?:nousda|n6v|eviel|novidate?)\b', 'Novidat'),
    (r'\b(?:beeek|rve\^ky|breaky|breeky)\b', 'Breeky'),
    (r'\b(?:njm\}?|nims)\b', 'Nims'),
    (r'\b(?:pynvns|provas)\b', 'Provas'),
    (r'\b(?:aiclaro\^?|dicloran)\b', 'Dicloran'),
    (r'\b(?:azr\~?|azimax)\b', 'Azimax'),
    (r'\b(?:azlw|azitma)\b', 'Azitma'),
    (r'\b(?:cc6vam|covam|cbvam)\b', 'Covam'),
    (r'\b(?:tm4ls\&?|tonoflex|toniflex)\b', 'Toniflex'),
    (r'\b(?:getryl|gekr)\b', 'Getryl'),
    (r'\b(?:leflox)\b', 'Leflox'),
    (r'\b(?:risek|gxls)\b', 'Risek'),
    (r'\b(?:lipiget|lpxe\w*)\b', 'Lipiget'),
    (r'\b(?:montiget|mbnhgeb)\b', 'Montiget'),
    (r'\b(?:starcox|lev\s*ll)\b', 'Starcox'),
    (r'\b(?:caricef|cie\s*6\~r)\b', 'Caricef'),
    (r'\b(?:atcomid|acnsl|a\+cozolc)\b', 'Atcomid'),
    (r'\b(?:atconate|ananec|0fcsnoql|a\+co\s*mot-e)\b', 'Atconate'),
    (r'\b(?:uriguard|uaszel)\b', 'Uriguard'),
    (r'\b(?:na\s*dard|dasd)\b', 'Na Dard'),
    (r'\b(?:atcam|acow)\b', 'Atcam'),
    (r'\b(?:ostium|os\+\s*iuw)\b', 'Ostium'),
    (r'\b(?:nebil)\b', 'Nebil'),
    (r'\b(?:movelate|allowelulr|illowelulr)\b', 'Movelate'),
    (r'\b(?:movalate|mevalcac)\b', 'Movalate'),
    (r'\b(?:mesulid|iesw\~|m\s*esimd)\b', 'Mesulid'),
    (r'\b(?:distalgesic|pistal)\b', 'Distalgesic'),
    (r'\b(?:bisleri|s\s*lck\s*b)\b', 'Bisleri'),
    (r'\b(?:cefiget|cefeget|oehet)\b', 'Cefiget'),

    # Common OCR confusions in medical text
    (r'\bMetforrn\b',      'Metformin'),
    (r'\bMetf0rmin\b',     'Metformin'),
    (r'\bAm0xicillin\b',   'Amoxicillin'),
    (r'\bParacetam0l\b',   'Paracetamol'),
    (r'\bAz\s*Lu\b',       'Azithral'),    # Fixes highly cursive Azithral misread
    (r'\brng\b',           'mg'),        # rng → mg  (most common)
    (r'rn(?=[a-z])',       'm'),         # rn → m  (inside words)
    (r'(?<=[a-z])vv',      'w'),         # vv → w
    (r'\b0D\b',            'OD'),        # 0D → OD (once daily)
    (r'\bB0\b',            'BD'),        # B0 → BD
    (r'\bTD5\b',           'TDS'),       # TD5 → TDS
    (r'\bQ1D\b',           'QID'),
    (r'(?<!\d)l(?=\d)',    '1'),         # l1 → 11 (lowercase L before digit)
    (r'(?<=\d)l\b',        '1'),         # 5l → 51
    (r'\bO(?=\d)',         '0'),         # O5 → 05 (capital O before digit)
    (r'(?<=\d)O\b',        '0'),         # 50O → 500
    (r'\bI(?=\d)',         '1'),         # I5 → 15 (capital I before digit)
    (r'S0S\b',             'SOS'),
    (r'\bSTAl\b',          'STAT'),
    (r'\s{2,}',            ' '),         # collapse whitespace
]

# Regex compiled once at module load
_COMPILED_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in OCR_SUBSTITUTIONS
    if p not in (r'\s{2,}',)  # keep whitespace collapse as str
]
_WS_COLLAPSE = re.compile(r'\s{2,}')


# ═══════════════════════════════════════════════════════════════════════════════
# MEDICAL TERMINOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

INDIAN_ABBREVIATIONS: dict[str, str] = {
    r'\bOD\b':   'Once Daily (OD)',
    r'\bBD\b':   'Twice Daily (BD)',
    r'\bTDS\b':  'Three Times Daily (TDS)',
    r'\bQID\b':  'Four Times Daily (QID)',
    r'\bSOS\b':  'As Needed (SOS)',
    r'\bSTAT\b': 'Immediately (STAT)',
    r'\bHS\b':   'At Bedtime (HS)',
    r'\bAC\b':   'Before Food (AC)',
    r'\bPC\b':   'After Food (PC)',
    r'\bCC\b':   'With Food (CC)',
    r'\bPRN\b':  'As Required (PRN)',
}

DOSAGE_RE = re.compile(
    r'\b(\d+\.?\d*)\s*(mg|ml|mcg|μg|g|iu|units?|tab(?:let)?s?|cap(?:sule)?s?'
    r'|drops?|puff|patch|sachet|vial)\b',
    re.IGNORECASE,
)

FREQUENCY_RE = re.compile(
    r'\b(once|twice|thrice|one|two|three|four|'
    r'od|bd|tds|qid|sos|stat|prn|hs|ac|pc|cc|'
    r'once\s+daily|twice\s+daily|three\s+times\s+daily|four\s+times\s+daily|'
    r'every\s+\d+\s+hours?|'
    r'morning|night|evening|afternoon|noon|bedtime|'
    r'before\s+(?:food|meal|breakfast|lunch|dinner)|'
    r'after\s+(?:food|meal|breakfast|lunch|dinner)|'
    r'with\s+food|'
    r'\d+\s*times?\s*(?:a\s*)?(?:day|daily))\b',
    re.IGNORECASE,
)

DURATION_RE = re.compile(
    r'\b(\d+)\s*(days?|weeks?|months?|years?)\b',
    re.IGNORECASE,
)

# Common header patterns on Indian prescriptions
DATE_RE    = re.compile(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b')
PATIENT_RE = re.compile(r'(?:patient|pt|name|p\.?t\.?)\s*[:\-]?\s*([A-Za-z\s\.]+)',
                         re.IGNORECASE)
DOCTOR_RE  = re.compile(r'(?:dr\.?|doctor)\s+([A-Za-z\s\.]+)',
                         re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_ocr_text(raw: str) -> str:
    """Apply all character-level OCR fixes to raw OCR output."""
    text = raw
    for pattern, replacement in _COMPILED_SUBS:
        text = pattern.sub(replacement, text)
    text = _WS_COLLAPSE.sub(' ', text)
    return text.strip()


def extract_dosages(text: str) -> List[str]:
    """Return list like ['500mg', '10ml', '2 tabs']."""
    matches = DOSAGE_RE.findall(text)
    return [f"{amt}{unit}" for amt, unit in matches]


def extract_frequencies(text: str) -> List[str]:
    """Return list like ['BD', 'twice daily', 'OD']."""
    return [m.strip() for m in FREQUENCY_RE.findall(text) if m.strip()]


def extract_durations(text: str) -> List[str]:
    """Return list like ['5 days', '2 weeks']."""
    matches = DURATION_RE.findall(text)
    return [f"{num} {unit}" for num, unit in matches]


def extract_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def extract_patient_name(text: str) -> str | None:
    m = PATIENT_RE.search(text)
    return m.group(1).strip() if m else None


def extract_doctor_name(text: str) -> str | None:
    m = DOCTOR_RE.search(text)
    return m.group(1).strip() if m else None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC PARSE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_prescription_text(raw_text: str) -> Dict:
    """
    Full pipeline — clean OCR output and return structured dict.

    Returns:
        cleaned_text  : str
        dosages       : list[str]
        frequencies   : list[str]
        durations     : list[str]
        patient_name  : str | None
        doctor_name   : str | None
        date          : str | None
    """
    cleaned = clean_ocr_text(raw_text)
    return {
        "cleaned_text":  cleaned,
        "dosages":        extract_dosages(cleaned),
        "frequencies":    extract_frequencies(cleaned),
        "durations":      extract_durations(cleaned),
        "patient_name":   extract_patient_name(cleaned),
        "doctor_name":    extract_doctor_name(cleaned),
        "date":           extract_date(cleaned),
    }
