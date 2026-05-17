"""
Pipeline de inferencia — Práctica 2 VCA
Uso: python menu.py
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from OCTDataset     import OCTDataset, EnhancedOCTDataset
from UNet           import UNet
from transforms     import CLAHE_CLIP_LIMIT
from evaluate_model import evaluate_model

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# --- Rutas del dataset de test ---
# IMAGE_PATH = os.path.join('OCT-test', 'images')
# MASK_PATH  = os.path.join('OCT-test', 'masks')
IMAGE_PATH = os.path.join('..', 'OCT-dataset', 'images')
MASK_PATH  = os.path.join('..', 'OCT-dataset', 'masks')

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# --- Experimentos ---
# Cada entrada: (nombre_display, arquitectura, ruta_pth, clahe)
# clahe: True si el modelo fue entrenado con CLAHE
EXPERIMENTS = {
    'A': ('Baseline — BCE',                    'unet', 'baseline.pth',      False),
    'B': ('Exp 1   — Dice Loss',               'unet', 'exp1_dice.pth',     False),
    'C': ('Exp 2   — BCE + Dice Loss',         'unet', 'exp2_bce_dice.pth', False),
    'D': ('Exp 3   — BCE + Dice Loss + CLAHE', 'unet', 'exp3_clahe.pth',    True),
    # Añadir aquí los siguientes experimentos:
    # 'E': ('Exp 4   — ...', 'unet', 'exp4_....pth', True),
}


def build_dataset(clahe):
    """Construye el dataset de test con el mismo preprocesado usado en entrenamiento."""
    if clahe:
        return EnhancedOCTDataset(IMAGE_PATH, MASK_PATH, clip_limit=CLAHE_CLIP_LIMIT)
    return OCTDataset(IMAGE_PATH, MASK_PATH)


def build_model(arch, pth, device):
    if arch == 'unet':
        model = UNet(input_channels=1, n_class=1)
    elif arch == 'resnet18':
        assert SMP_AVAILABLE, "Instala segmentation-models-pytorch para usar este modelo."
        model = smp.Unet(encoder_name='resnet18', encoder_weights=None,
                         in_channels=1, classes=1)
    else:
        raise ValueError(f'Arquitectura desconocida: {arch}')
    model.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
    model.to(device)
    return model


def ask(prompt, valid_options):
    while True:
        choice = input(prompt).strip().upper()
        if choice in valid_options:
            return choice
        print(f"  Opcion no valida. Elige entre: {', '.join(valid_options)}")


def print_separator():
    print('=' * 55)


def menu_model():
    print_separator()
    print('  Modelo a evaluar')
    print_separator()
    for key, (name, _, pth, clahe) in EXPERIMENTS.items():
        full_pth = os.path.join(MODELS_DIR, pth)
        status   = 'OK' if os.path.isfile(full_pth) else 'NO ENCONTRADO'
        flags    = f'CLAHE={clahe}'
        print(f'  [{key}] {name}  [{status}]  ({flags})')
    print('  [T] Todos los modelos')
    print('  [Q] Salir')
    print_separator()
    return ask('Elige modelo: ', list(EXPERIMENTS.keys()) + ['T', 'Q'])


def run_experiment(key, device):
    name, arch, pth_file, clahe = EXPERIMENTS[key]
    pth = os.path.join(MODELS_DIR, pth_file)
    if not os.path.isfile(pth):
        print(f'\n  ERROR: no se encuentra {pth}')
        return
    print(f'\nCargando {name}...')
    ds     = build_dataset(clahe)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    model  = build_model(arch, pth, device)
    evaluate_model(model, loader, device, experiment_name=name)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    if not os.path.isdir(IMAGE_PATH) or not os.path.isdir(MASK_PATH):
        print(f'\nERROR: no se encuentra el dataset de test.')
        print(f'  Esperado en: {IMAGE_PATH}  y  {MASK_PATH}')
        print(f'  Coloca las imagenes del profesor en esas rutas y vuelve a ejecutar.')
        sys.exit(1)

    while True:
        choice = menu_model()
        if choice == 'Q':
            print('Saliendo.')
            break
        elif choice == 'T':
            for key in EXPERIMENTS:
                run_experiment(key, device)
        else:
            run_experiment(choice, device)


if __name__ == '__main__':
    main()
