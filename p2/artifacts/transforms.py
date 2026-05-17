import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

# --- Parámetros de preprocesado ---
CLAHE_CLIP_LIMIT = 2.0
RSIZE            = (416, 624)

# Rellenar con los valores obtenidos en el entrenamiento final
TRAIN_MEAN = None  # TODO: añadir valor
TRAIN_STD  = None  # TODO: añadir valor

# --- CLAHE ---
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))


def apply_clahe(img):
    """Aplica CLAHE sobre imagen numpy uint8 en escala de grises."""
    return _clahe.apply(img)


# --- Transform base (sin augmentation) ---
# Se aplica en val y test
def get_base_transform():
    assert TRAIN_MEAN is not None and TRAIN_STD is not None, \
        "Rellena TRAIN_MEAN y TRAIN_STD en transforms.py antes de ejecutar."
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RSIZE, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[TRAIN_MEAN], std=[TRAIN_STD]),
    ])
