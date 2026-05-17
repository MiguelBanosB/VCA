import torch
import matplotlib.pyplot as plt


def get_segmentation_masks(outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    return (probs > threshold) * 1.0


def show_result(orig, gt, prediction, title=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [orig, gt, prediction, orig * prediction]
    titles = ['Original', 'Ground Truth', 'Prediccion', 'Overlap']
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(t)
        ax.axis('off')
    if title is not None:
        fig.suptitle(title, fontweight='bold')
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
