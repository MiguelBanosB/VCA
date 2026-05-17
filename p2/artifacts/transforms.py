import cv2

# --- Parámetros de preprocesado ---
CLAHE_CLIP_LIMIT = 2.0
RSIZE            = (416, 624)

# --- CLAHE ---
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))


def apply_clahe(img):
    """Aplica CLAHE sobre imagen numpy uint8 en escala de grises."""
    return _clahe.apply(img)
