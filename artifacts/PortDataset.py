import os
import numpy as np
import pandas as pd

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

class PortDataset(Dataset):
    def __init__(self, df, label_col='ship', transform=None):
        # Si la tarea es Docked/Undocked, filtramos solo imágenes con barco
        # (una imagen sin barco no puede estar atracada ni no atracada)
        if label_col == 'docked':
            self.df = df[df['ship'] == 1].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        # Columna del DataFrame que contiene la etiqueta (ship o docked)
        self.label_col = label_col
        # Transformaciones a aplicar (resize, normalización, augmentation...)
        self.transform = transform

    def __len__(self):
        # PyTorch necesita saber el tamaño del dataset para crear los batches
        return len(self.df)

    def __getitem__(self, idx):
        # Obtenemos la fila correspondiente al índice
        row = self.df.iloc[idx]

        # Cargamos la imagen y la convertimos a RGB (por si alguna es escala de grises)
        img = Image.open(row['filepath']).convert('RGB')

        # Extraemos la etiqueta como tensor float (0.0 o 1.0) para BCEWithLogitsLoss
        label = torch.tensor(row[self.label_col], dtype=torch.float32)

        # Aplicamos las transformaciones si las hay
        if self.transform:
            img = self.transform(img)

        return img, label