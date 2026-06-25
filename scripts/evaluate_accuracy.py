"""
Evaluate the trained ResNet18 model on the BD dataset's held-out test set.

This is the definitive accuracy evaluation — clean cropped word images,
same distribution as training data.

Usage:
    python scripts/evaluate_accuracy.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz


def evaluate_model_on_bd_testset():
    """Evaluate ResNet18 classifier directly on clean cropped word images."""
    from model.predict import get_classifier

    clf = get_classifier()
    if not clf.is_loaded:
        print("ERROR: Model weights not found. Run: python scripts/train_model.py")
        return

    # Find BD dataset test split
    search_root = Path.home() / "Downloads"
    candidates = list(search_root.rglob("testing_labels.csv"))

    test_csv = None
    for c in candidates:
        potential_root = c.parent.parent
        if (potential_root / "Training").exists() and (potential_root / "Validation").exists():
            test_csv = c
            break

    if test_csv is None:
        print("Could not find testing_labels.csv. Skipping model evaluation.")
        return

    test_words_dir = test_csv.parent / "testing_words"
    df = pd.read_csv(test_csv)

    print("=" * 55)
    print("  MODEL EVALUATION ON BD DATASET TEST SET")
    print("=" * 55)
    print(f"Test images : {len(df)}")
    print(f"Classes     : {df['MEDICINE_NAME'].nunique()}")
    print()

    import torch
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    correct_exact = 0
    correct_fuzzy = 0
    top3_correct  = 0
    total         = 0

    for _, row in df.iterrows():
        img_path = test_words_dir / row["IMAGE"]
        if not img_path.exists():
            continue

        true_label = row["MEDICINE_NAME"]
        img = Image.open(img_path).convert("RGB")

        # Top-1 prediction
        pred_label, conf = clf.predict(img)
        total += 1

        if pred_label == true_label:
            correct_exact += 1
            top3_correct  += 1
        else:
            # Fuzzy match (handles slight spelling differences)
            if fuzz.ratio(pred_label.lower(), true_label.lower()) > 80:
                correct_fuzzy += 1

            # Top-3 check
            tensor = transform(img).unsqueeze(0).to(clf.device)
            with torch.no_grad():
                logits = clf._model(tensor)
                top3_idxs = logits[0].topk(3).indices.tolist()
            top3_labels = [clf._labels.get(i, "") for i in top3_idxs]
            if true_label in top3_labels:
                top3_correct += 1

    if total == 0:
        print("No test images found.")
        return

    exact_acc = correct_exact / total * 100
    fuzzy_acc = (correct_exact + correct_fuzzy) / total * 100
    top3_acc  = top3_correct  / total * 100

    print(f"Total tested    : {total}")
    print(f"Top-1 Accuracy  : {exact_acc:.1f}%")
    print(f"Fuzzy Accuracy  : {fuzzy_acc:.1f}%  (80% string-match threshold)")
    print(f"Top-3 Accuracy  : {top3_acc:.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    evaluate_model_on_bd_testset()
