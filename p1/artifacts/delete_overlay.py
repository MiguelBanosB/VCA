import numpy as np
from PIL import Image


def delete_overlay(img, edge_w=3, threshold=5):
    arr = np.array(img.convert('L'))
    h, w = arr.shape

    def is_overlay_row(i):
        return arr[i, :edge_w].mean() < threshold and arr[i, -edge_w:].mean() < threshold

    # Banda superior: avanzamos fila a fila hasta la primera fila sin overlay
    top_cut = 0
    for i in range(h):
        if is_overlay_row(i):
            top_cut = i + 1
        else:
            break

    # Banda inferior: retrocedemos fila a fila desde el final
    bottom_cut = h
    for i in range(h - 1, -1, -1):
        if is_overlay_row(i):
            bottom_cut = i
        else:
            break

    if top_cut == 0 and bottom_cut == h:
        return img  # Sin overlay, devolvemos la imagen original

    return img.crop((0, top_cut, w, bottom_cut))
