import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             recall_score, precision_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)


def evaluate_model(model, loader, device, label_names=('No-Ship', 'Ship'), experiment_name='Modelo'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu().tolist()
            preds  = [int(p >= 0.5) for p in probs]
            all_preds.extend(preds)
            all_labels.extend(labels.long().tolist())
            all_probs.extend(probs)

    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')

    print(f"\n{experiment_name} · Resultados Test")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Recall    : {rec:.4f}  <- métrica prioritaria")
    print(f"  Precision : {prec:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=label_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, colorbar=False)
    ax.set_title(f'{experiment_name} — Matriz de confusión (Test)', fontweight='bold')
    plt.tight_layout()
    plt.show()

    return acc, f1, rec, prec, auc
