# 💊 Indian Doctor Prescription OCR & Safety Checker

> *"Built because my own family members have been given wrong medicines due to illegible prescriptions."*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transfer%20Learning-red?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)](https://streamlit.io)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-Deep%20Learning-4CAF50)](https://github.com/JaidedAI/EasyOCR)

---

## 🩺 The Problem

Indian doctors have notoriously illegible handwriting. A chemist misreading **"Metformin"** as **"Metronidazole"** gives a completely different drug — a diabetes patient instead receives an antibiotic. **This happens thousands of times daily across India.**

This problem is solved in the US (Epic, Cerner) but **completely unsolved in India** where 90% of prescriptions are still handwritten.

---

## 🚀 What It Does

1. **📸 Upload** any handwritten or printed prescription photo
2. **🔍 OCR** — CRAFT word detector + ResNet18 classifier (with EasyOCR fallback) extracts medicine names
3. **🧹 Clean** — 30+ OCR error corrections specific to medical text
4. **💊 Match** — Fuzzy token matching against 246,000+ Indian medicines database
5. **⚠️ Warn** — Detects dangerous look-alike drug name confusions (57 known look-alike pairs)

---

## 🏗️ Architecture
prescription-ocr/

│

├── app.py                    ← Streamlit web frontend

│

├── core/                     ← Core ML/NLP pipeline

│   ├── ocr_engine.py         ← Hybrid OCR (CRAFT + ResNet18 Classifier + EasyOCR fallback)

│   ├── text_cleaner.py       ← Medical OCR error correction

│   ├── medicine_matcher.py   ← Fuzzy medicine matching

│   └── danger_checker.py     ← Look-alike drug safety warnings

│

├── model/                    ← Transfer learning pipeline

│   ├── train.py              ← ResNet18 fine-tuning on BD dataset

│   ├── predict.py            ← Model inference (ResNet18 classifier)

│   └── weights/              ← Trained model weights and label map

│

├── data/

│   └── medicines.csv         ← 246,000+ Indian medicine database

│

├── scripts/                  ← Runner and evaluation scripts

│   ├── train_model.py        ← Script to train the classifier

│   └── evaluate_accuracy.py  ← Script to evaluate classifier on test set

│

└── tests/                    ← pytest test suite

├── test_text_cleaner.py

├── test_medicine_matcher.py

└── test_danger_checker.py

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit frontend
```bash
streamlit run app.py
```

### 3. Run tests
```bash
pytest tests/ -v
```

---

## 🧠 ML Training

1. Download and extract the Doctor's Handwritten Prescription BD dataset to your `Downloads` folder.
2. Run the training script:
```bash
python scripts/train_model.py
```
3. Run the evaluation script to check performance on the held-out test set:
```bash
python scripts/evaluate_accuracy.py
```

**Dataset:** Doctors Handwritten Prescription BD dataset (4,680 cropped word images across 78 classes).

**Results on held-out test set (780 images):**

| Metric | Score |
|---|---|
| Top-1 Accuracy | 79.1% |
| Top-3 Accuracy | 87.9% |
| Test Images | 780 (10 per class) |

**Model:** ResNet18 (pretrained on ImageNet)
- Phase 1: Train classification head only (backbone frozen)
- Phase 2: Unfreeze backbone for full fine-tuning
- Augmentation: rotation, color jitter, affine shift, grayscale (simulates phone photo variations)

---

## 🔬 Technical Highlights

| Component | Technique | Why |
|---|---|---|
| **OCR (Primary)** | CRAFT + ResNet18 Classifier | Fine-tuned on 78 BD medicine classes — 79.1% Top-1 accuracy |
| **Fallback OCR** | EasyOCR (CRAFT + CRNN) | General-purpose text recognition for unmatched words |
| **Preprocessing** | Denoising + Adaptive Thresholding | Enhances text structure for detection |
| **Medicine Matching** | RapidFuzz (Token Sort Ratio) | Handles OCR typos & misspellings |
| **Transfer Learning** | ResNet18 ImageNet → Prescription | 10x faster than training from scratch |
| **Danger Detection** | Levenshtein distance | Finds similar-sounding drug names |

---

## 💊 Sample Dangerous Drug Pairs Detected

| Drug A | Drug B | Risk |
|---|---|---|
| Metformin | Metronidazole | 🔴 HIGH — diabetes drug vs antibiotic |
| Amlodipine | Amitriptyline | 🔴 HIGH — BP drug vs antidepressant |
| Glimepiride | Glipizide | 🔴 HIGH — two different diabetes drugs |
| Hydroxychloroquine | Chloroquine | 🔴 HIGH — dosage-critical confusion |
| Cetirizine | Levocetirizine | 🟡 MODERATE — antihistamine confusion |

---

## 🛠️ Tech Stack

`Python 3.10` · `PyTorch` · `ResNet18` · `EasyOCR` · `OpenCV` · `Streamlit` · `RapidFuzz` · `Pandas` · `scikit-learn`

---

## 👤 Built By

**Aditya Joshi**
B.Tech CSE (AI & ML)
CMR College of Engineering and Technology, Hyderabad

---

## ⚠️ Disclaimer

This tool assists in identifying medicines — it does **NOT** replace medical advice.
Always verify with your doctor or licensed pharmacist before consuming any medicine.

---

## 🔮 Conclusions & Next Steps

This project successfully proves the complete architectural pipeline required for a medical safety checker (Preprocessing → OCR → Regex Cleaning → Fuzzy Matching). However, end-to-end testing on highly cursive real-world Indian prescriptions revealed a significant limitation: the ResNet18 classifier achieves **79.1% Top-1 accuracy on clean cropped word images** from its training distribution, but performance degrades sharply on unconstrained real prescription photos where CRAFT detects word regions that fall outside the 78 trained classes.

### Why does real-world accuracy drop?

The ResNet18 was trained on the BD dataset — clean, cropped, well-lit word images of 78 specific medicine names. Real prescriptions contain hundreds of different drug names, doctor annotations, and highly cursive writing that the classifier has never seen. When the classifier confidence is low, the pipeline falls back to EasyOCR — which itself struggles with Indian doctor handwriting (e.g., extracting `"Az Lu T Sn"` from `"tab Azithral 500"`). No amount of downstream fuzzy matching can reconstruct a string if the OCR extraction is garbage.

### The Technical Roadmap to Production

To reach 90%+ accuracy on real prescriptions:

1. **Custom HTR Fine-Tuning:** Fine-tune **Microsoft TrOCR** or a custom CRNN specifically on the `Doctors prescriptions handwriting.v1i.coco` dataset — a full handwriting recognition model, not just a word classifier.
2. **Domain-Specific Language Model:** Add a small NLP correction layer after OCR that uses a medical dictionary prior to improve character-level predictions before fuzzy matching.

---
*Built to solve a real-world problem. Code is open-sourced.*