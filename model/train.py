"""
Train a ResNet18 classifier on the Doctor's Handwritten Prescription BD dataset.

Dataset: 4,680 cropped word images across 78 medicine name classes
Architecture: ResNet18 (ImageNet pre-trained) → fine-tuned 78-class head
Strategy: Transfer learning — backbone frozen for first 3 epochs, then full fine-tune

Usage:
    python scripts/train_model.py
    # or directly:
    python -m model.train
"""
from __future__ import annotations

import os
import sys
import json
import time
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS — auto-discover dataset under Downloads
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR   = Path(__file__).parent
WEIGHTS_DIR = MODEL_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = WEIGHTS_DIR / "best_model.pth"
LABEL_MAP_PATH  = WEIGHTS_DIR / "label_map.json"


def find_dataset_root() -> Path:
    """Auto-discover the BD dataset under Downloads.

    Looks for training_labels.csv and verifies the expected folder
    structure (Training/Validation/Testing) exists at the parent.
    Prefers the 'archive_3' path to avoid picking up other datasets.
    """
    search_root = Path.home() / "Downloads"
    candidates = list(search_root.rglob("training_labels.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find 'training_labels.csv' anywhere under "
            f"{search_root}. Make sure the dataset is extracted there."
        )

    # Sort to prefer archive_3 path
    def preference(p: Path) -> int:
        s = str(p).lower()
        if "archive_3" in s:
            return 0
        if "bd dataset" in s or "prescription bd" in s:
            return 1
        return 2

    candidates.sort(key=preference)

    for csv_path in candidates:
        # training_labels.csv → Training/ → dataset_root/
        potential_root = csv_path.parent.parent
        if (
            (potential_root / "Training").exists()
            and (potential_root / "Validation").exists()
            and (potential_root / "Testing").exists()
        ):
            return potential_root

    # Last resort: use the first found
    return candidates[0].parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class MedicineWordDataset(Dataset):
    """
    Loads cropped handwritten medicine word images with class labels.

    Directory layout expected:
        <split>/
            <split>_labels.csv     ← columns: IMAGE, MEDICINE_NAME, GENERIC_NAME
            <split>_words/
                0.png
                1.png
                ...
    """

    def __init__(self, split_dir: Path, split_name: str,
                 class_to_idx: dict, transform):
        csv_path  = split_dir / f"{split_name}_labels.csv"
        self.img_dir   = split_dir / f"{split_name}_words"
        self.transform = transform
        self.class_to_idx = class_to_idx

        df = pd.read_csv(csv_path)
        # Keep only rows whose image file actually exists
        self.records = [
            (row["IMAGE"], row["MEDICINE_NAME"])
            for _, row in df.iterrows()
            if (self.img_dir / row["IMAGE"]).exists()
        ]

        print(f"  [{split_name}] {len(self.records)} images, "
              f"{len(set(r[1] for r in self.records))} classes")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        filename, label = self.records[idx]
        img = Image.open(self.img_dir / filename).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def make_transforms(phase: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                    scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 with custom classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device,
              train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    epochs: int      = 10,
    batch_size: int  = 32,
    lr: float        = 1e-3,
    unfreeze_epoch: int = 4,   # epoch at which to unfreeze the backbone
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Training on: {device}")

    # ── Dataset discovery ────────────────────────────────────────────────────
    dataset_root = find_dataset_root()
    print(f"[*] Dataset root: {dataset_root}\n")

    train_dir = dataset_root / "Training"
    val_dir   = dataset_root / "Validation"
    test_dir  = dataset_root / "Testing"

    # Build label map from training CSV
    train_csv  = pd.read_csv(train_dir / "training_labels.csv")
    classes    = sorted(train_csv["MEDICINE_NAME"].unique().tolist())
    generics   = (
        train_csv.drop_duplicates("MEDICINE_NAME")
                 .set_index("MEDICINE_NAME")["GENERIC_NAME"]
                 .to_dict()
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes  = len(classes)
    print(f"[*] Classes: {num_classes}")

    # Save label map immediately so inference can use it even if training stops
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "generic_lookup": generics,
            "num_classes": num_classes,
        }, f, indent=2)
    print(f"[*] Label map saved -> {LABEL_MAP_PATH}\n")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("[*] Loading datasets...")
    train_ds = MedicineWordDataset(train_dir, "training",   class_to_idx, make_transforms("train"))
    val_ds   = MedicineWordDataset(val_dir,   "validation", class_to_idx, make_transforms("val"))
    test_ds  = MedicineWordDataset(test_dir,  "testing",    class_to_idx, make_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = build_model(num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    print(f"\n[*] Starting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):

        # Phase 2: unfreeze backbone for fine-tuning
        if epoch == unfreeze_epoch:
            print(">>> Unfreezing backbone for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - unfreeze_epoch + 1
            )

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            marker = "  <- best"

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"{elapsed:.1f}s{marker}")

    # ── Test set evaluation ──────────────────────────────────────────────────
    print(f"\n[*] Best val acc: {best_val_acc:.3f}")
    print("[*] Evaluating on held-out test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    _, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"[+] Test accuracy: {test_acc:.3f}")
    print(f"\n[+] Done! Model saved to {BEST_MODEL_PATH}")
    return best_val_acc, test_acc


if __name__ == "__main__":
    train()
