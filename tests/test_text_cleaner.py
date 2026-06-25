"""
Tests for core/text_cleaner.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from core.text_cleaner import (
    clean_ocr_text, extract_dosages, extract_frequencies,
    extract_durations, parse_prescription_text,
)


class TestCleanOcrText:
    def test_fixes_rng_to_mg(self):
        assert "500mg" in clean_ocr_text("Metformin 500rng OD")

    def test_collapses_whitespace(self):
        assert "Metformin OD" == clean_ocr_text("Metformin   OD")

    def test_no_change_for_clean_text(self):
        text = "Paracetamol 500mg BD"
        assert clean_ocr_text(text) == text


class TestExtractDosages:
    def test_single_dosage(self):
        assert "500mg" in extract_dosages("Metformin 500mg")

    def test_multiple_dosages(self):
        result = extract_dosages("Metformin 500mg BD, Amlodipine 5mg OD")
        assert len(result) == 2

    def test_ml_dosage(self):
        assert "10ml" in extract_dosages("Syrup 10ml TDS")

    def test_no_dosage(self):
        assert extract_dosages("Take this medicine") == []


class TestExtractFrequencies:
    def test_bd(self):
        assert any("BD" in f.upper() for f in extract_frequencies("Metformin 500mg BD"))

    def test_od(self):
        assert any("OD" in f.upper() for f in extract_frequencies("Atenolol 25mg OD"))

    def test_tds(self):
        assert any("TDS" in f.upper() for f in extract_frequencies("Augmentin 625mg TDS"))

    def test_twice_daily(self):
        freqs = extract_frequencies("Take twice daily")
        assert any("twice" in f.lower() for f in freqs)


class TestExtractDurations:
    def test_days(self):
        assert "5 days" in extract_durations("Take for 5 days")

    def test_weeks(self):
        assert "2 weeks" in extract_durations("Course: 2 weeks")

    def test_no_duration(self):
        assert extract_durations("Metformin 500mg OD") == []


class TestParsePrescriptionText:
    def test_full_pipeline(self):
        result = parse_prescription_text("Metformin 500rng BD for 30 days")
        assert result["cleaned_text"] != ""
        assert len(result["dosages"]) >= 1
        assert len(result["frequencies"]) >= 1
        assert len(result["durations"]) >= 1

    def test_returns_all_keys(self):
        result = parse_prescription_text("Paracetamol 500mg OD")
        for key in ("cleaned_text", "dosages", "frequencies", "durations",
                    "patient_name", "doctor_name", "date"):
            assert key in result
