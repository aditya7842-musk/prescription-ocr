"""
Tests for core/medicine_matcher.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from core.medicine_matcher import find_medicines_in_text, search_medicines


class TestFindMedicinesInText:
    def test_finds_exact_name(self):
        results = find_medicines_in_text("Metformin 500mg BD")
        assert any("metformin" in r["matched_name"].lower() or "metformin" in r["generic_name"].lower() for r in results)

    def test_finds_misspelled_name(self):
        # Metforrn is a common OCR output for Metformin
        results = find_medicines_in_text("Metforrn 500mg BD")
        assert any("metformin" in r["matched_name"].lower() or "metformin" in r["generic_name"].lower() for r in results)

    def test_finds_multiple_medicines(self):
        text = "Metformin 500mg BD, Amlodipine 5mg OD, Azithromycin 500mg OD"
        results = find_medicines_in_text(text)
        assert len(results) >= 2

    def test_no_duplicates(self):
        text = "Metformin 500mg BD Metformin 250mg OD"
        results = find_medicines_in_text(text)
        matched = [r["matched_name"].lower() for r in results]
        assert len(matched) == len(set(matched))

    def test_result_has_required_fields(self):
        results = find_medicines_in_text("Paracetamol 500mg")
        if results:
            r = results[0]
            for field in ("found_name", "matched_name", "generic_name",
                          "category", "match_score", "confidence_tier"):
                assert field in r

    def test_match_score_in_range(self):
        results = find_medicines_in_text("Augmentin 625mg TDS")
        for r in results:
            assert 0 <= r["match_score"] <= 100

    def test_confidence_tier_valid(self):
        results = find_medicines_in_text("Metformin 500mg")
        for r in results:
            assert r["confidence_tier"] in ("High", "Medium", "Low")

    def test_empty_text(self):
        results = find_medicines_in_text("")
        assert results == []

    def test_nonsense_text(self):
        results = find_medicines_in_text("xxxxxxxx yyyyyyy zzzzzzzz")
        assert isinstance(results, list)


class TestSearchMedicines:
    def test_returns_results(self):
        results = search_medicines("Metformin")
        assert len(results) > 0

    def test_fuzzy_search(self):
        results = search_medicines("Amlodpin")  # misspelled
        assert any("amlodipine" in r["matched_name"].lower() or "amlodipine" in r["generic_name"].lower() for r in results)

    def test_limit_respected(self):
        results = search_medicines("met", limit=3)
        assert len(results) <= 3
