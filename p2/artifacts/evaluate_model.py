import torch
import matplotlib.pyplot as plt
import numpy as np


def get_segmentation_masks(outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    return (probs > threshold) * 1.0


def show_result(orig, gt, prediction, title=None):
    """
    Muestra 4 paneles: imagen original, máscara ground truth, predicción de la red
    y overlay sobre la imagen original donde verde indica detección correcta (TP)
    y rojo indica error (FP o FN).
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    ax = axes.ravel()

    orig_np = orig.numpy() if hasattr(orig, 'numpy') else np.array(orig)
    gt_np   = (gt.numpy() if hasattr(gt, 'numpy') else np.array(gt)) > 0.5
    pred_np = (prediction.numpy() if hasattr(prediction, 'numpy') else np.array(prediction)) > 0.5

    ax[0].imshow(orig_np, cmap='gray')
    ax[0].set_title('Orig')
    ax[0].axis('off')

    ax[1].imshow(gt_np, cmap='gray')
    ax[1].set_title('GT')
    ax[1].axis('off')

    ax[2].imshow(pred_np, cmap='gray')
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    overlay = np.stack([orig_np, orig_np, orig_np], axis=-1)
    overlay[gt_np & pred_np]    = [0.0, 1.0, 0.0]
    overlay[(~gt_np) & pred_np] = [1.0, 0.0, 0.0]
    overlay[gt_np & (~pred_np)] = [1.0, 0.0, 0.0]

    ax[3].imshow(overlay)
    ax[3].set_title('Overlay (verde=TP, rojo=FP/FN)')
    ax[3].axis('off')

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, loader, device, experiment_name='Modelo', n_vis=3):
    model.eval()

    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = get_segmentation_masks(model(images))
            tp += (preds * masks).sum().item()
            fp += (preds * (1 - masks)).sum().item()
            fn += ((1 - preds) * masks).sum().item()
            tn += ((1 - preds) * (1 - masks)).sum().item()

    eps  = 1e-8
    iou  = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    acc  = (tp + tn) / (tp + fp + fn + tn + eps)

    print(f'\n=== {experiment_name} — Resultados en Test ===')
    print(f'  IoU       : {iou:.4f}')
    print(f'  Dice      : {dice:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  Accuracy  : {acc:.4f}')

    # Visualización de n_vis muestras
    shown = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = get_segmentation_masks(model(images))
            for i in range(len(images)):
                if shown >= n_vis:
                    break
                show_result(
                    images[i].cpu().squeeze(),
                    masks[i].cpu().squeeze(),
                    preds[i].cpu().squeeze(),
                    title=f'{experiment_name} — Muestra {shown + 1}'
                )
                shown += 1
            if shown >= n_vis:
                break

    return {'iou': iou, 'dice': dice, 'precision': prec, 'recall': rec, 'accuracy': acc}
