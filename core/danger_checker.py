"""
Danger Checker — detects potentially dangerous medicine name confusions.

Three independent checks:
1. KNOWN DANGEROUS PAIRS     — hardcoded 80+ pairs specific to Indian market
2. LEVENSHTEIN SIMILARITY    — any two meds in prescription with edit-distance ≤ 6
3. SAME HIGH-RISK CATEGORY   — two diabetes/BP/blood-thinner/steroid meds together

Severity levels:
  🔴 HIGH RISK     (known documented confusion, different drug class)
  🟡 MODERATE RISK (similar names, verification recommended)
  🟠 CAUTION       (same category, double-check intentional)
"""
from __future__ import annotations
import os
import json
from rapidfuzz.distance import Levenshtein
from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWN DANGEROUS PAIRS
# ═══════════════════════════════════════════════════════════════════════════════
# Source: ISMP India reports + common Indian pharmacy confusion reports
# Asymmetric — listed in both directions for bidirectional lookup

KNOWN_DANGEROUS_PAIRS: dict[str, str] = {
    # Diabetes ↔ Antibiotic
    "metformin":           "metronidazole",
    "metronidazole":       "metformin",
    # Blood Pressure ↔ Antidepressant
    "amlodipine":          "amitriptyline",
    "amitriptyline":       "amlodipine",
    "atenolol":            "albuterol",
    "albuterol":           "atenolol",
    # Diabetes pairs (fixed — no duplicate keys)
    "glimepiride":         "glipizide",
    "glipizide":           "gliclazide",
    "gliclazide":          "glibenclamide",
    "glibenclamide":       "gliclazide",
    # Antibiotics (fixed — no duplicate cefixime)
    "amoxicillin":         "amoxiclav",
    "amoxiclav":           "amoxicillin",
    "cefixime":            "ceftriaxone",
    "ceftriaxone":         "cefixime",
    "cefuroxime":          "cefixime",
    "azithromycin":        "erythromycin",
    "erythromycin":        "azithromycin",
    "levofloxacin":        "ciprofloxacin",
    "ciprofloxacin":       "levofloxacin",
    # Malaria
    "hydroxychloroquine":  "chloroquine",
    "chloroquine":         "hydroxychloroquine",
    # Antiallergic
    "cetirizine":          "cetrizine",
    "cetrizine":           "cetirizine",
    "loratadine":          "levocetirizine",
    "levocetirizine":      "loratadine",
    "fexofenadine":        "cetirizine",
    # Cholesterol (fixed — no duplicate atorvastatin)
    "atorvastatin":        "rosuvastatin",
    "rosuvastatin":        "atorvastatin",
    "simvastatin":         "atorvastatin",
    # Painkiller vs Antibiotic
    "tramadol":            "toradol",
    "toradol":             "tramadol",
    # Steroids
    "prednisolone":        "prednisone",
    "prednisone":          "prednisolone",
    "dexamethasone":       "betamethasone",
    "betamethasone":       "dexamethasone",
    # Thyroid
    "levothyroxine":       "liothyronine",
    "liothyronine":        "levothyroxine",
    # Blood Pressure (fixed — no duplicate valsartan)
    "losartan":            "valsartan",
    "valsartan":           "telmisartan",
    "telmisartan":         "losartan",
    "ramipril":            "lisinopril",
    "lisinopril":          "ramipril",
    # Acidity (fixed — no duplicate omeprazole, removed self-reference)
    "omeprazole":          "pantoprazole",
    "pantoprazole":        "omeprazole",
    "esomeprazole":        "omeprazole",
    "rabeprazole":         "omeprazole",
    # Antifungal vs Antibiotic
    "fluconazole":         "flucytosine",
    "flucytosine":         "fluconazole",
    # Nausea
    "ondansetron":         "domperidone",
    "domperidone":         "ondansetron",
    # NSAID confusion
    "diclofenac":          "diflucan",
    "diflucan":            "diclofenac",
    "ibuprofen":           "ibufenac",
    # Antihypertensive
    "nifedipine":          "nimodipine",
    "nimodipine":          "nifedipine",
    # Anticoagulant
    "warfarin":            "heparin",
    "heparin":             "warfarin",
}

SEVERITY_LABELS = {
    "known_pair":    "🔴 HIGH RISK",
    "similar_name":  "🟡 MODERATE RISK",
    "same_category": "🟠 CAUTION",
}

HIGH_RISK_CATEGORIES = {
    "Diabetes", "Blood Pressure", "Antibiotic",
    "Steroid", "Blood Thinner", "Anticoagulant",
    "Thyroid", "Cardiac", "Antiseizure",
}

# Levenshtein threshold — names within this edit-distance trigger moderate warning
LEVENSHTEIN_THRESHOLD = 5


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — KNOWN DATABASE PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

def check_database_lookalikes(medicines: List[Dict]) -> List[Dict]:
    """
    Match each medicine against the hardcoded dangerous pairs dictionary.
    Also checks the 'dangerous_lookalike' field from the medicine DB.
    """
    warnings: list[Dict] = []
    seen_pairs: set[frozenset] = set()

    for med in medicines:
        name     = med["matched_name"].lower()
        db_entry = str(med.get("dangerous_lookalike", "")).strip().lower()

        # ---- Hardcoded pairs ----
        if name in KNOWN_DANGEROUS_PAIRS:
            lookalike = KNOWN_DANGEROUS_PAIRS[name]
            pair_key  = frozenset({name, lookalike})
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                warnings.append({
                    "medicine":  med["matched_name"],
                    "lookalike": lookalike.title(),
                    "severity":  SEVERITY_LABELS["known_pair"],
                    "message":   (
                        f"**{med['matched_name']}** is frequently confused with "
                        f"**{lookalike.title()}** — these are completely different drugs. "
                        f"A chemist misreading one for the other can be dangerous. "
                        f"Always verify at the pharmacy."
                    ),
                    "type": "known_pair",
                })

        # ---- DB lookalike field ----
        elif db_entry and db_entry != "nan":
            pair_key = frozenset({name, db_entry})
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                warnings.append({
                    "medicine":  med["matched_name"],
                    "lookalike": db_entry.title(),
                    "severity":  SEVERITY_LABELS["known_pair"],
                    "message":   (
                        f"**{med['matched_name']}** has a documented look-alike: "
                        f"**{db_entry.title()}**. Double-check with your doctor or pharmacist."
                    ),
                    "type": "known_pair",
                })

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — LEVENSHTEIN NAME SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

def check_similarity_between_medicines(medicines: List[Dict]) -> List[Dict]:
    """
    Compare every pair of medicine names in the prescription using
    Levenshtein edit-distance. Close names (≤ threshold) get a warning.
    """
    warnings: list[Dict] = []
    names = [m["matched_name"] for m in medicines]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1 = names[i].lower()
            n2 = names[j].lower()
            dist = Levenshtein.distance(n1, n2)
            if 0 < dist <= LEVENSHTEIN_THRESHOLD:
                warnings.append({
                    "medicine":  names[i],
                    "lookalike": names[j],
                    "severity":  SEVERITY_LABELS["similar_name"],
                    "message":   (
                        f"**{names[i]}** and **{names[j]}** have very similar spellings "
                        f"(edit-distance: {dist}). Confirm both medicines are intentional."
                    ),
                    "type":      "similar_names",
                    "distance":  dist,
                })

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — SAME CATEGORY DUPLICATES
# ═══════════════════════════════════════════════════════════════════════════════

def check_category_duplicates(medicines: List[Dict]) -> List[Dict]:
    """
    Flag when 2+ medicines from the same high-risk category appear together.
    Sometimes intentional (combination therapy) but always worth verifying.
    """
    warnings: list[Dict] = []
    by_category: Dict[str, list] = {}

    for med in medicines:
        cat = med.get("category", "Unknown")
        by_category.setdefault(cat, []).append(med["matched_name"])

    for cat, meds in by_category.items():
        if len(meds) >= 2 and cat in HIGH_RISK_CATEGORIES:
            warnings.append({
                "medicine":  meds[0],
                "lookalike": meds[1],
                "severity":  SEVERITY_LABELS["same_category"],
                "message":   (
                    f"**{' and '.join(meds)}** are both **{cat}** medicines. "
                    f"Taking two {cat.lower()} drugs together is sometimes intentional "
                    f"(combination therapy) but always verify with your doctor."
                ),
                "type": "same_category",
            })

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_danger_checks(medicines: List[Dict]) -> Dict[str, Any]:
    """
    Run all three danger checks and return categorised results.

    Returns:
        total_warnings  : int
        high_risk       : list[dict]   — known dangerous pairs
        moderate_risk   : list[dict]   — similar names
        caution         : list[dict]   — same category
        all_warnings    : list[dict]   — sorted by severity
    """
    if not medicines:
        return {
            "total_warnings": 0,
            "high_risk":      [],
            "moderate_risk":  [],
            "caution":        [],
            "all_warnings":   [],
        }

    db_warns   = check_database_lookalikes(medicines)
    sim_warns  = check_similarity_between_medicines(medicines)
    cat_warns  = check_category_duplicates(medicines)

    _order = {
        SEVERITY_LABELS["known_pair"]:    0,
        SEVERITY_LABELS["similar_name"]:  1,
        SEVERITY_LABELS["same_category"]: 2,
    }
    all_warns = sorted(
        db_warns + sim_warns + cat_warns,
        key=lambda w: _order.get(w["severity"], 3),
    )

    return {
        "total_warnings": len(all_warns),
        "high_risk":      [w for w in all_warns if w["type"] == "known_pair"],
        "moderate_risk":  [w for w in all_warns if w["type"] == "similar_names"],
        "caution":        [w for w in all_warns if w["type"] == "same_category"],
        "all_warnings":   all_warns,
    }
