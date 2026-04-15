"""
Pipeline de inferencia — Práctica 1 VCA
Uso: python menu.py
"""
import os
import sys
import torch
from torch.utils.data import DataLoader

from artifacts.load_dataset      import load_dataset
from artifacts.PortDataset       import PortDataset
from  artifacts.BaseCNN           import BaseCNN
from  artifacts.build_pretrained  import build_pretrained
from  artifacts.transforms        import transform_base
from  artifacts.evaluate_model    import evaluate_model

#  Rutas — ajusta si es necesario 
IMAGE_PATH = os.path.join('..', 'P1-Material', 'images')
SHIP_CSV   = os.path.join('..', 'P1-Material', 'ship.csv')
DOCKED_CSV = os.path.join('..', 'P1-Material', 'docked.csv')

MODELS = {
    'A': {
        'name':      'Experimento A — BaseCNN sin augmentation',
        'arch':      'base',
        'pth':       os.path.join('..', 'model_A.pth'),
        'task':      'ship',
        'labels':    ('No-Ship', 'Ship'),
    },
    'B': {
        'name':      'Experimento B — BaseCNN con augmentation',
        'arch':      'base',
        'pth':       os.path.join('..', 'model_B.pth'),
        'task':      'ship',
        'labels':    ('No-Ship', 'Ship'),
    },
    'C': {
        'name':      'Experimento C — ResNet18 sin augmentation',
        'arch':      'pretrained',
        'pth':       os.path.join('..', 'model_C.pth'),
        'task':      'ship',
        'labels':    ('No-Ship', 'Ship'),
    },
    'D': {
        'name':      'Experimento D — ResNet18 con augmentation',
        'arch':      'pretrained',
        'pth':       os.path.join('..', 'model_D.pth'),
        'task':      'ship',
        'labels':    ('No-Ship', 'Ship'),
    },
}


def build_model(arch, pth, device):
    model = BaseCNN() if arch == 'base' else build_pretrained()
    state = torch.load(pth, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def print_menu():
    print("\n" + "=" * 50)
    print("  Pipeline de Inferencia — Práctica 1 VCA")
    print("=" * 50)
    for key, cfg in MODELS.items():
        pth_exists = os.path.isfile(cfg['pth'])
        status = "OK" if pth_exists else "NO ENCONTRADO"
        print(f"  [{key}] {cfg['name']}  [{status}]")
    print("  [T] Evaluar TODOS los modelos")
    print("  [Q] Salir")
    print("=" * 50)


def run_inference(key, df, device):
    cfg = MODELS[key]

    if not os.path.isfile(cfg['pth']):
        print(f"\n  ERROR: no se encuentra {cfg['pth']}")
        return

    print(f"\nCargando {cfg['name']}...")
    model  = build_model(cfg['arch'], cfg['pth'], device)
    loader = DataLoader(
        PortDataset(df, label_col=cfg['task'], transform=transform_base),
        batch_size=32,
        shuffle=False,
    )
    evaluate_model(model, loader, device,
                   label_names=cfg['labels'],
                   experiment_name=cfg['name'])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\nCargando dataset de test...")
    df = load_dataset(IMAGE_PATH, SHIP_CSV, DOCKED_CSV)
    print(f"  {len(df)} imágenes cargadas.")

    while True:
        print_menu()
        choice = input("\nElige una opción: ").strip().upper()

        if choice == 'Q':
            print("Saliendo.")
            break
        elif choice == 'T':
            for key in MODELS:
                run_inference(key, df, device)
        elif choice in MODELS:
            run_inference(choice, df, device)
        else:
            print("  Opción no válida, intenta de nuevo.")


if __name__ == '__main__':
    main()
