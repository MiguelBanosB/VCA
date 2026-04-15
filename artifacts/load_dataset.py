import os
import numpy as np
import pandas as pd

def load_dataset(image_path, ship_csv, docked_csv):
    """
    Carga y fusiona los CSVs de etiquetas, construye las rutas completas
    y devuelve un DataFrame listo para ser usado por PortDataset.
    
    Parámetros:
        image_path  : ruta a la carpeta de imágenes
        ship_csv    : ruta al CSV con etiquetas ship/no-ship
        docked_csv  : ruta al CSV con etiquetas docked/undocked
    
    Retorna:
        df : DataFrame con columnas [filename, ship, docked, filepath]
             docked = -1 para imágenes sin barco
    """
    # Carga ambos CSVs
    ship_df   = pd.read_csv(ship_csv,   sep=';', header=0, names=['filename', 'ship'])
    docked_df = pd.read_csv(docked_csv, sep=';', header=0, names=['filename', 'docked'])

    # Fusiona por nombre de fichero — left join para conservar todas las imágenes
    df = ship_df.merge(docked_df, on='filename', how='left')

    # Imágenes sin barco no tienen etiqueta docked — se marcan con -1
    df['docked'] = df['docked'].fillna(-1).astype(int)

    # Construye la ruta completa a cada imagen
    df['filepath'] = df['filename'].apply(lambda f: os.path.join(image_path, f))

    return df