"""
Pipeline de inferencia — Práctica 1 VCA
Uso: python menu.py
"""
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from load_dataset     import load_dataset
from PortDataset      import PortDataset
from BaseCNN          import BaseCNN
from build_pretrained import build_pretrained
from transforms       import transform_base
from evaluate_model   import evaluate_model

# ── Rutas — ajusta si es necesario ───────────────────────────────────────────
IMAGE_PATH = os.path.join('P1-Material', 'images')
SHIP_CSV   = os.path.join('P1-Material', 'ship.csv')
DOCKED_CSV = os.path.join('P1-Material', 'docked.csv')

# ── Configuración de experimentos ─────────────────────────────────────────────
# Cada entrada: (nombre_display, arquitectura, pth_con_overlay, pth_sin_overlay)
EXPERIMENTS = {
    'ship': {
        'task':   'ship',
        'labels': ('No-Ship', 'Ship'),
        'models': {
            'A': ('BaseCNN sin augmentation',  'base',       'models/ship_models/model_A.pth',       'models/ship_models/model_A_without_ov.pth'),
            'B': ('BaseCNN con augmentation',  'base',       'models/ship_models/model_B.pth',       'models/ship_models/model_B_without_ov.pth'),
            'C': ('ResNet18 sin augmentation', 'pretrained', 'models/ship_models/model_C.pth',       'models/ship_models/model_C_without_ov.pth'),
            'D': ('ResNet18 con augmentation', 'pretrained', 'models/ship_models/model_D.pth',       'models/ship_models/model_D_without_ov.pth'),
        }
    },
    'docked': {
        'task':   'docked',
        'labels': ('Undocked', 'Docked'),
        'models': {
            'A': ('BaseCNN sin augmentation',  'base',       'models/docked_models/model_A_docked.pth',       'models/docked_models/model_A_docked_without_ov.pth'),
            'B': ('BaseCNN con augmentation',  'base',       'models/docked_models/model_B_docked.pth',       'models/docked_models/model_B_docked_without_ov.pth'),
            'C': ('ResNet18 sin augmentation', 'pretrained', 'models/docked_models/model_C_docked.pth',       'models/docked_models/model_C_docked_without_ov.pth'),
            'D': ('ResNet18 con augmentation', 'pretrained', 'models/docked_models/model_D_docked.pth',       'models/docked_models/model_D_docked_without_ov.pth'),
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────


def pth_path(filename):
    return filename


def build_model(arch, pth, device):
    model = BaseCNN() if arch == 'base' else build_pretrained()
    model.load_state_dict(torch.load(pth, map_location=device))
    model.to(device)
    return model


def ask(prompt, valid_options):
    while True:
        choice = input(prompt).strip().upper()
        if choice in valid_options:
            return choice
        print(f"  Opción no válida. Elige entre: {', '.join(valid_options)}")


def print_separator():
    print("=" * 55)


def menu_task():
    print_separator()
    print("  Tarea de clasificación")
    print_separator()
    print("  [S] Ship / No-Ship")
    print("  [D] Docked / Undocked")
    print("  [Q] Salir")
    print_separator()
    return ask("Elige tarea: ", ['S', 'D', 'Q'])


def menu_model(task_cfg):
    print_separator()
    print("  Modelo")
    print_separator()
    for key, (name, _, pth_ov, pth_no_ov) in task_cfg['models'].items():
        ok_ov  = "OK" if os.path.isfile(pth_path(pth_ov))    else "NO ENCONTRADO"
        ok_nov = "OK" if os.path.isfile(pth_path(pth_no_ov)) else "NO ENCONTRADO"
        print(f"  [{key}] {name}  [con OV: {ok_ov} | sin OV: {ok_nov}]")
    print("  [T] Todos los modelos")
    print("  [B] Volver")
    print_separator()
    return ask("Elige modelo: ", list(task_cfg['models'].keys()) + ['T', 'B'])


def menu_overlay():
    print_separator()
    print("  Dataset")
    print_separator()
    print("  [1] Con overlay")
    print("  [2] Sin overlay")
    print("  [B] Volver")
    print_separator()
    return ask("Elige dataset: ", ['1', '2', 'B'])


def run_experiment(task_cfg, model_key, overlay, df, device):
    name, arch, pth_ov, pth_no_ov = task_cfg['models'][model_key]
    pth = pth_path(pth_ov if overlay == '1' else pth_no_ov)
    ov_label = "con overlay" if overlay == '1' else "sin overlay"
    exp_name = f"Exp {model_key} — {name} ({ov_label})"

    if not os.path.isfile(pth):
        print(f"\n  ERROR: no se encuentra {pth}")
        return

    print(f"\nCargando {exp_name}...")
    model  = build_model(arch, pth, device)
    loader = DataLoader(
        PortDataset(df, label_col=task_cfg['task'], transform=transform_base,
                    remove_overlay=(overlay == '2')),
        batch_size=32, shuffle=False,
    )
    evaluate_model(model, loader, device,
                   label_names=task_cfg['labels'],
                   experiment_name=exp_name)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\nCargando dataset de test...")
    df = load_dataset(IMAGE_PATH, SHIP_CSV, DOCKED_CSV)
    print(f"  {len(df)} imágenes cargadas.")

    while True:
        task_choice = menu_task()
        if task_choice == 'Q':
            print("Saliendo.")
            break

        task_cfg = EXPERIMENTS['ship' if task_choice == 'S' else 'docked']

        while True:
            model_choice = menu_model(task_cfg)
            if model_choice == 'B':
                break

            while True:
                ov_choice = menu_overlay()
                if ov_choice == 'B':
                    break

                if model_choice == 'T':
                    for key in task_cfg['models']:
                        run_experiment(task_cfg, key, ov_choice, df, device)
                else:
                    run_experiment(task_cfg, model_choice, ov_choice, df, device)

                break  # vuelve al menú de modelo tras evaluar


if __name__ == '__main__':
    main()
