"""
Model Inference — loads the trained ResNet18 classifier and classifies
a single cropped word image into one of the 78 medicine name classes.

Usage:
    from model.predict import MedicineClassifier
    clf = MedicineClassifier()
    label, confidence = clf.predict(pil_image)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np

MODEL_DIR       = Path(__file__).parent
WEIGHTS_DIR     = MODEL_DIR / "weights"
BEST_MODEL_PATH = WEIGHTS_DIR / "best_model.pth"
LABEL_MAP_PATH  = WEIGHTS_DIR / "label_map.json"

_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MedicineClassifier:
    """
    Singleton-friendly wrapper around the trained ResNet18.
    Lazily loads model weights on first call to predict().
    """

    _instance: Optional["MedicineClassifier"] = None

    def __init__(self):
        self._model  = None
        self._labels: dict[str, str] = {}  # idx_to_class
        self._generics: dict[str, str] = {}  # medicine_name → generic
        self.num_classes = 0
        self.is_loaded   = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> bool:
        """Load model weights and label map. Returns True if successful."""
        if self.is_loaded:
            return True

        if not BEST_MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
            return False

        try:
            with open(LABEL_MAP_PATH) as f:
                data = json.load(f)

            self._labels   = {int(k): v for k, v in data["idx_to_class"].items()}
            self._generics = data.get("generic_lookup", {})
            self.num_classes = data["num_classes"]

            # Re-build same architecture as train.py
            from torchvision import models as tvm
            model = tvm.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, self.num_classes),
            )
            model.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location=self.device)
            )
            model.eval()
            self._model  = model.to(self.device)
            self.is_loaded = True
            print(f"[+] Medicine classifier loaded ({self.num_classes} classes)")
            return True

        except Exception as e:
            print(f"⚠️  Could not load classifier: {e}")
            return False

    def predict(self, image) -> tuple[str, float]:
        """
        Classify a word image.

        Args:
            image: PIL Image or numpy array (BGR or RGB)

        Returns:
            (medicine_name, confidence_0_to_1)
        """
        if not self.is_loaded:
            if not self.load():
                return "", 0.0

        # Accept numpy arrays (e.g. OpenCV crop)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image)
            else:
                # OpenCV uses BGR
                image = Image.fromarray(image[:, :, ::-1])

        tensor = _TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        label = self._labels.get(idx.item(), "")
        return label, round(conf.item(), 3)

    def predict_batch(self, images: list) -> list[tuple[str, float]]:
        """Classify multiple images efficiently in one forward pass."""
        if not self.is_loaded:
            if not self.load():
                return [("", 0.0)] * len(images)

        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = Image.fromarray(img)
                else:
                    img = Image.fromarray(img[:, :, ::-1])
            tensors.append(_TRANSFORM(img))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = self._model(batch)
            probs  = torch.softmax(logits, dim=1)
            confs, idxs = probs.max(dim=1)

        return [
            (self._labels.get(idx.item(), ""), round(conf.item(), 3))
            for idx, conf in zip(idxs, confs)
        ]

    def get_generic(self, medicine_name: str) -> str:
        """Lookup generic name for a classified medicine."""
        return self._generics.get(medicine_name, "")


# Module-level singleton
_classifier: Optional[MedicineClassifier] = None


def get_classifier() -> MedicineClassifier:
    """Return the shared classifier instance (loads lazily)."""
    global _classifier
    if _classifier is None:
        _classifier = MedicineClassifier()
        _classifier.load()
    return _classifier
