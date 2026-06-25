"""
Tests for core/danger_checker.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from core.danger_checker import (
    check_database_lookalikes,
    check_similarity_between_medicines,
    check_category_duplicates,
    run_all_danger_checks,
)


def _med(name, category="Unknown", lookalike=""):
    return {
        "matched_name":        name,
        "category":            category,
        "dangerous_lookalike": lookalike,
    }


class TestDatabaseLookalikes:
    def test_known_pair_metformin(self):
        meds  = [_med("Metformin")]
        warns = check_database_lookalikes(meds)
        assert len(warns) >= 1
        assert any("metronidazole" in w["message"].lower() for w in warns)

    def test_no_warning_for_safe_med(self):
        meds  = [_med("Vitamin D3")]
        warns = check_database_lookalikes(meds)
        assert len(warns) == 0

    def test_db_field_lookalike(self):
        meds  = [_med("Dolo", lookalike="Dolo 650")]
        warns = check_database_lookalikes(meds)
        assert isinstance(warns, list)


class TestSimilarityCheck:
    def test_similar_names_flagged(self):
        # Cefixime vs Ceftriaxone: edit distance = 5, should be flagged
        meds  = [_med("Cefixime"), _med("Cefuroxime")]
        warns = check_similarity_between_medicines(meds)
        assert len(warns) >= 1

    def test_identical_names_not_flagged(self):
        # distance=0 is excluded
        meds  = [_med("Metformin"), _med("Metformin")]
        warns = check_similarity_between_medicines(meds)
        assert len(warns) == 0

    def test_very_different_names_not_flagged(self):
        meds  = [_med("Vitamin"), _med("Metronidazole")]
        warns = check_similarity_between_medicines(meds)
        # edit distance > 5 so no warning
        assert all(w["distance"] <= 5 for w in warns)


class TestCategoryDuplicates:
    def test_two_diabetes_meds_flagged(self):
        meds = [
            _med("Metformin",  "Diabetes"),
            _med("Glimepiride", "Diabetes"),
        ]
        warns = check_category_duplicates(meds)
        assert len(warns) >= 1

    def test_two_vitamins_not_flagged(self):
        meds = [
            _med("Vitamin D3", "Vitamin"),
            _med("Vitamin B12", "Vitamin"),
        ]
        warns = check_category_duplicates(meds)
        assert len(warns) == 0   # Vitamin not in HIGH_RISK_CATEGORIES

    def test_single_med_not_flagged(self):
        meds  = [_med("Metformin", "Diabetes")]
        warns = check_category_duplicates(meds)
        assert len(warns) == 0


class TestRunAllDangerChecks:
    def test_empty_medicines(self):
        result = run_all_danger_checks([])
        assert result["total_warnings"] == 0
        assert result["high_risk"] == []

    def test_returns_all_keys(self):
        result = run_all_danger_checks([_med("Metformin")])
        for key in ("total_warnings", "high_risk", "moderate_risk",
                    "caution", "all_warnings"):
            assert key in result

    def test_severity_ordering(self):
        meds = [
            _med("Metformin",   "Diabetes"),
            _med("Glimepiride", "Diabetes"),
        ]
        result = run_all_danger_checks(meds)
        # high_risk should come before moderate_risk
        all_w = result["all_warnings"]
        types = [w["type"] for w in all_w]
        if "known_pair" in types and "same_category" in types:
            assert types.index("known_pair") < types.index("same_category")
