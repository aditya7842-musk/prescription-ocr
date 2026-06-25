"""
OCR Engine — hybrid pipeline for handwritten prescription recognition.

Strategy (in order of preference):
  1. CRAFT detector + ResNet18 classifier  → best for the 78 BD medicine classes
  2. CRAFT detector + EasyOCR text         → fallback: use what EasyOCR actually reads
  3. EasyOCR on the preprocessed image     → last resort if CRAFT finds nothing
"""
import os
import cv2
import numpy as np


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Improve image quality before OCR.
    Upscale + denoise + adaptive threshold → better CRAFT detection
    on small or blurry prescription handwriting.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def extract_text_from_image(image_path: str) -> dict:
    """
    Main entry point — extract medicine names from a prescription image.

    Runs the best available pipeline and always returns raw OCR text
    so the fuzzy matcher downstream can work even when classifier confidence
    is low.
    """
    if not os.path.exists(image_path):
        return {"raw_text": "", "lines": [], "confidence": 0.0,
                "error": "Image file not found"}

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > 10:
        return {"raw_text": "", "lines": [], "confidence": 0.0,
                "error": f"Image too large ({file_size_mb:.1f} MB). Please use an image under 10 MB."}

    return _run_pipeline(image_path)


def _run_pipeline(image_path: str) -> dict:
    """
    Three-stage pipeline:
      Stage 1 — CRAFT + ResNet18: classify each word crop (best for BD classes)
      Stage 2 — CRAFT + EasyOCR text: use what EasyOCR actually read if classifier fails
      Stage 3 — Plain EasyOCR: fallback on preprocessed whole image
    Results from all stages are merged so we get both the classifier output
    AND the raw OCR text for fuzzy matching.
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        # Load ResNet18 classifier (lazy singleton — returns None if not trained)
        clf = None
        try:
            from model.predict import get_classifier
            clf = get_classifier()
            if not clf.is_loaded:
                clf = None
        except Exception:
            clf = None

        # ── Stage 1 & 2: CRAFT detection ────────────────────────────────────
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"raw_text": "", "lines": [], "confidence": 0.0,
                    "error": "Cannot read image"}

        # Scale up small images for better CRAFT detection
        h, w = img_bgr.shape[:2]
        scale = 2.0 if max(h, w) < 1000 else 1.0
        img_scaled = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_CUBIC)

        craft_results = reader.readtext(img_scaled)

        classifier_labels = []   # from ResNet18
        easyocr_words     = []   # from EasyOCR text recognition

        if craft_results:
            from PIL import Image as PILImage

            crops = []
            for (bbox, easyocr_text, easyocr_conf) in craft_results:
                # Collect EasyOCR's own text reading
                word = easyocr_text.strip()
                if word and len(word) >= 2 and easyocr_conf > 0.1:
                    easyocr_words.append(word)

                # Crop the region for ResNet classifier
                pts = np.array(bbox, dtype=np.int32)
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                pad = 4
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img_scaled.shape[1], x2 + pad)
                y2 = min(img_scaled.shape[0], y2 + pad)
                crop = img_scaled[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(PILImage.fromarray(crop[:, :, ::-1]))

            # Stage 1: ResNet18 classification
            if clf is not None and crops:
                predictions = clf.predict_batch(crops)
                seen = set()
                for (label, conf) in predictions:
                    if label and conf > 0.85 and label not in seen:
                        classifier_labels.append(label)
                        seen.add(label)

        # ── Merge results ────────────────────────────────────────────────────
        # Priority: classifier labels first (they are actual medicine names),
        # then raw EasyOCR words (useful for fuzzy matching in medicine_matcher)
        combined_lines = []
        seen_lower = set()

        for label in classifier_labels:
            key = label.lower()
            if key not in seen_lower:
                combined_lines.append(label)
                seen_lower.add(key)

        for word in easyocr_words:
            key = word.lower()
            if key not in seen_lower:
                combined_lines.append(word)
                seen_lower.add(key)

        # ── Stage 3: plain EasyOCR fallback if both above produced nothing ──
        if not combined_lines:
            try:
                processed = preprocess_image(image_path)
                plain_results = reader.readtext(processed)
                for (_, text, conf) in plain_results:
                    word = text.strip()
                    if word and len(word) >= 2 and conf > 0.05:
                        combined_lines.append(word)
            except Exception:
                pass

        if not combined_lines:
            return {"raw_text": "", "lines": [], "confidence": 0.0,
                    "error": "No text detected in image"}

        raw_text = " ".join(combined_lines)
        # Confidence: higher if classifier found something, lower if only EasyOCR
        confidence = 0.75 if classifier_labels else 0.40

        return {
            "raw_text":   raw_text,
            "lines":      combined_lines,
            "confidence": round(confidence, 2),
            "engine":     "craft+resnet18" if classifier_labels else "easyocr",
        }

    except Exception as e:
        return {"raw_text": "", "lines": [], "confidence": 0.0, "error": str(e)}