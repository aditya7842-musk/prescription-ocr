"""
Medicine Matcher — finds medicine names in OCR text using fuzzy matching.
Uses 246,068 Indian medicine database.

Fixed to use actual CSV columns:
  - 'name'               → brand name
  - 'short_composition1' → generic name (with dosage stripped)
  - 'Is_discontinued'    → filter out discontinued medicines
"""
import os
import re
import pandas as pd
from rapidfuzz import fuzz, process
from typing import List, Dict, Any

# Go up one level from core/ to reach root data/ folder
DATA_PATH       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medicines.csv")
MATCH_THRESHOLD = 60

_db         = None
_full_names  = None
_short_list  = None


def clean_generic_name(composition: str) -> str:
    """
    Strip dosage info from composition.
    'Amoxycillin  (500mg)' → 'Amoxycillin'
    """
    if not composition or pd.isna(composition):
        return ''
    return re.sub(r'\s*\([^)]*\)', '', str(composition)).strip()


def load_medicine_database() -> pd.DataFrame:
    """Load and cache medicines CSV."""
    global _db
    if _db is not None:
        return _db

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Medicine database not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, dtype=str).fillna('')

    # Append custom sample prescription medicines to ensure they match correctly
    CUSTOM_MEDICINES = [
        {"name": "Novidat 500mg Tablet", "short_composition1": "Ciprofloxacin", "Is_discontinued": "False"},
        {"name": "Getryl 2mg Tablet", "short_composition1": "Glimepiride", "Is_discontinued": "False"},
        {"name": "Covam 5/80mg Tablet", "short_composition1": "Valsartan + Amlodipine", "Is_discontinued": "False"},
        {"name": "Fexet 120mg Tablet", "short_composition1": "Fexofenadine", "Is_discontinued": "False"},
        {"name": "Starcox 90mg Tablet", "short_composition1": "Etoricoxib", "Is_discontinued": "False"},
        {"name": "Leflox 250mg Tablet", "short_composition1": "Levofloxacin", "Is_discontinued": "False"},
        {"name": "Cefiget 400mg Capsule", "short_composition1": "Cefixime", "Is_discontinued": "False"},
        {"name": "Risek 20mg Capsule", "short_composition1": "Omeprazole", "Is_discontinued": "False"},
        {"name": "Lipiget 10mg Tablet", "short_composition1": "Atorvastatin", "Is_discontinued": "False"},
        {"name": "Montiget 10mg Tablet", "short_composition1": "Montelukast", "Is_discontinued": "False"},
        {"name": "Distalgesic Tablet", "short_composition1": "Dextropropoxyphene + Paracetamol", "Is_discontinued": "False"},
        {"name": "Atcomid 100mg Tablet", "short_composition1": "Lacosamide", "Is_discontinued": "False"},
        {"name": "Atconate 35mg Tablet", "short_composition1": "Risedronate Sodium", "Is_discontinued": "False"},
        {"name": "Mesulid 100mg Tablet", "short_composition1": "Nimesulide", "Is_discontinued": "False"},
        {"name": "Movelate Gel", "short_composition1": "Mucopolysaccharide polysulphate + Salicylic acid", "Is_discontinued": "False"},
        {"name": "Uriguard Tablet", "short_composition1": "Flavoxate", "Is_discontinued": "False"},
        {"name": "Na Dard Tablet", "short_composition1": "Pain Relief OTC", "Is_discontinued": "False"},
        {"name": "Atcam Tablet", "short_composition1": "Lornoxicam", "Is_discontinued": "False"},
        {"name": "My-D Tablet", "short_composition1": "Cholecalciferol", "Is_discontinued": "False"},
        {"name": "Ostium Tablet", "short_composition1": "Calcium + Vitamin D", "Is_discontinued": "False"},
        {"name": "Nebil Tablet", "short_composition1": "Nebivolol", "Is_discontinued": "False"},
        {"name": "Provas IV Injection", "short_composition1": "Paracetamol", "Is_discontinued": "False"},
        {"name": "Caricef 400mg Capsule", "short_composition1": "Cefixime", "Is_discontinued": "False"},
        {"name": "Azimax 500mg Tablet", "short_composition1": "Azithromycin", "Is_discontinued": "False"},
        {"name": "Toniflex Tablet", "short_composition1": "Tramadol + Paracetamol", "Is_discontinued": "False"},
        {"name": "Nims Tablet", "short_composition1": "Nimesulide", "Is_discontinued": "False"},
        {"name": "Dicloran Tablet", "short_composition1": "Diclofenac", "Is_discontinued": "False"},
        {"name": "Breeky Tablet", "short_composition1": "Misoprostol", "Is_discontinued": "False"},
        {"name": "Pronaz Tablet", "short_composition1": "Naproxen", "Is_discontinued": "False"},           
        {"name": "Movax Tablet", "short_composition1": "Tizanidine", "Is_discontinued": "False"},
        
    ]
    custom_df = pd.DataFrame(CUSTOM_MEDICINES).fillna('')
    df = pd.concat([df, custom_df], ignore_index=True)

    # Filter out discontinued medicines
    df = df[df['Is_discontinued'].str.lower() != 'true'].copy()

    # Use actual column names from the CSV
    df['brand_name']   = df['name'].str.strip()
    df['generic_name'] = df['short_composition1'].apply(clean_generic_name)

    # Derive category from generic name
    def get_category(comp):
        c = str(comp).lower()
        if any(x in c for x in ['azithromycin','amoxycillin','ciprofloxacin','ceftriaxone','cefixime','metronidazole','levofloxacin']):
            return 'Antibiotic'
        if any(x in c for x in ['paracetamol','ibuprofen','aceclofenac','diclofenac','nimesulide']):
            return 'Painkiller'
        if any(x in c for x in ['metformin','glimepiride','insulin','glipizide','vildagliptin','sitagliptin']):
            return 'Diabetes'
        if any(x in c for x in ['amlodipine','atenolol','telmisartan','losartan','ramipril','olmesartan']):
            return 'Blood Pressure'
        if any(x in c for x in ['atorvastatin','rosuvastatin','simvastatin']):
            return 'Cholesterol'
        if any(x in c for x in ['omeprazole','pantoprazole','rabeprazole','esomeprazole','ranitidine']):
            return 'Acidity'
        if any(x in c for x in ['cetirizine','fexofenadine','loratadine','hydroxyzine','chlorpheniramine']):
            return 'Antiallergic'
        if any(x in c for x in ['salbutamol','montelukast','budesonide','fluticasone']):
            return 'Respiratory'
        if any(x in c for x in ['amitriptyline','alprazolam','lorazepam','escitalopram','sertraline']):
            return 'Psychiatric'
        if any(x in c for x in ['vitamin','calcium','iron','folic','zinc','magnesium']):
            return 'Supplement'
        return 'General'

    df['category'] = df['short_composition1'].apply(get_category)

    # Known dangerous pairs
    KNOWN_PAIRS = {
        'metformin':          'Metronidazole',
        'metronidazole':      'Metformin',
        'amlodipine':         'Amitriptyline',
        'amitriptyline':      'Amlodipine',
        'glimepiride':        'Glipizide',
        'glipizide':          'Glimepiride',
        'hydroxychloroquine': 'Chloroquine',
        'cefixime':           'Ceftriaxone',
        'atorvastatin':       'Rosuvastatin',
        'cetirizine':         'Cetrizine',
    }

    def get_lookalike(comp):
        c = str(comp).lower()
        for key, val in KNOWN_PAIRS.items():
            if key in c:
                return val
        return ''

    df['dangerous_lookalike'] = df['short_composition1'].apply(get_lookalike)

    _db = df
    return _db


def get_name_lists():
    """Build lookup lists for fuzzy matching."""
    global _full_names, _short_list
    if _full_names is not None:
        return _full_names, _short_list

    db = load_medicine_database()

    full_names    = db['brand_name'].tolist()
    generic_names = db['generic_name'].tolist()

    # Short names = first word of brand name (what doctors actually write)
    short_names = [n.split()[0] if n.split() else n for n in full_names]

    _full_names = full_names
    _short_list = short_names + generic_names

    return _full_names, _short_list


def split_into_candidates(text: str) -> List[str]:
    """Split OCR text into word candidates."""
    words      = text.split()
    candidates = []

    for word in words:
        clean = word.strip('.,;:!?()[]{}"\'-')
        if len(clean) >= 3:
            candidates.append(clean)

    for i in range(len(words) - 1):
        pair = f"{words[i].strip('.,;')} {words[i+1].strip('.,;')}".strip()
        if len(pair) >= 5:
            candidates.append(pair)

    return list(set(candidates))


def find_medicines_in_text(text: str) -> List[Dict[str, Any]]:
    """
    Main function — scan OCR text and find medicine names.
    Returns list of matched medicines with details.
    """
    if not text or not text.strip():
        return []

    db = load_medicine_database()
    full_names, search_list = get_name_lists()

    candidates      = split_into_candidates(text)
    raw_matches = []

    for candidate in candidates:
        if len(candidate) < 3:
            continue

        result = process.extractOne(
            candidate,
            search_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=MATCH_THRESHOLD,
        )

        if result is None:
            continue

        matched_name, score, idx = result

        # Map back to full brand name
        half = len(full_names)
        if idx < half:
            brand_name = full_names[idx]
        else:
            generic_idx = idx - half
            brand_name  = full_names[generic_idx] if generic_idx < len(full_names) else matched_name

        # Look up full details
        medicine_row = db[db['brand_name'] == brand_name]
        if medicine_row.empty:
            first_word   = brand_name.split()[0]
            medicine_row = db[db['brand_name'].str.startswith(first_word)]
        if medicine_row.empty:
            continue

        row = medicine_row.iloc[0]

        # Calculate confidence tier
        score_val = round(score)
        if score_val >= 90:
            tier = "High"
        elif score_val >= 75:
            tier = "Medium"
        else:
            tier = "Low"

        raw_matches.append({
            "found_name":          candidate,
            "matched_name":        row['brand_name'],
            "generic_name":        row['generic_name'],
            "category":            row['category'],
            "match_score":         score_val,
            "confidence_tier":     tier,
            "dangerous_lookalike": row.get('dangerous_lookalike', ''),
        })

    # Sort raw matches by score descending
    raw_matches.sort(key=lambda x: x['match_score'], reverse=True)

    # Dedup by first word
    found_medicines = []
    seen_names      = set()
    for m in raw_matches:
        brand_name = m["matched_name"]
        dedup_key = brand_name.split()[0].lower() if brand_name.split() else brand_name.lower()
        if dedup_key not in seen_names:
            seen_names.add(dedup_key)
            found_medicines.append(m)

    return found_medicines[:10]


def search_medicines(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for medicines in the database using fuzzy matching on brand/generic names."""
    if not query or not query.strip():
        return []

    db = load_medicine_database()
    full_names, search_list = get_name_lists()

    # Find the top matches in search_list
    results = process.extract(
        query,
        search_list,
        scorer=fuzz.token_sort_ratio,
        limit=limit * 2,  # get more for deduping
    )

    if not results:
        return []

    found = []
    seen = set()

    for matched_name, score, idx in results:
        half = len(full_names)
        if idx < half:
            brand_name = full_names[idx]
        else:
            generic_idx = idx - half
            brand_name  = full_names[generic_idx] if generic_idx < len(full_names) else matched_name

        dedup_key = brand_name.split()[0].lower() if brand_name.split() else brand_name.lower()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        medicine_row = db[db['brand_name'] == brand_name]
        if medicine_row.empty:
            first_word   = brand_name.split()[0]
            medicine_row = db[db['brand_name'].str.startswith(first_word)]
        if medicine_row.empty:
            continue

        row = medicine_row.iloc[0]

        # Calculate confidence tier
        score_val = round(score)
        if score_val >= 90:
            tier = "High"
        elif score_val >= 75:
            tier = "Medium"
        else:
            tier = "Low"

        found.append({
            "found_name":          query,
            "matched_name":        row['brand_name'],
            "generic_name":        row['generic_name'],
            "category":            row['category'],
            "match_score":         score_val,
            "confidence_tier":     tier,
            "dangerous_lookalike": row.get('dangerous_lookalike', ''),
        })

        if len(found) >= limit:
            break

    return found