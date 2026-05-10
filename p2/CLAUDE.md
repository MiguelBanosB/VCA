# P2 — Segmentación de fluido patológico en imágenes OCT

## Enunciado

### Contexto clínico
La Tomografía de Coherencia Óptica (OCT) es una técnica no invasiva que obtiene imágenes volumétricas de alta resolución de los tejidos oculares. Las imágenes OCT muestran en detalle las capas retinianas y permiten detectar alteraciones como la presencia de **fluido patológico**, relacionado con diversas enfermedades oculares.

El fluido patológico aparece como **zonas oscuras redondeadas dentro de la retina** — la luz atraviesa el líquido sin reflejarse, por eso aparece hiporreflectivo (oscuro). La máscara marca esas zonas en **blanco sobre negro**.

La variación de forma entre retinas (oval vs plana) se debe principalmente a la posición del corte: los cortes centrales pasan por la fóvea (elevación característica) y dan perfil más ovalado; los periféricos son más planos. También influye levemente la anatomía individual.

### Objetivo
Desarrollo de una metodología para la segmentación automática de regiones de fluido patológico en imágenes OCT.

### Dataset
- **50 imágenes** OCT en escala de grises (`.jpg`) con sus máscaras binarias correspondientes
- Directorio: `p2/OCT-dataset/images/` y `p2/OCT-dataset/masks/`
- Nombres: `sample_01.jpg` a `sample_50.jpg`
- Las imágenes son panorámicas de la retina en sección transversal
- **Desbalanceo severo**: el fluido ocupa una fracción pequeña del total de píxeles (2-10% en casos severos, casi nada en casos leves como sample_25)
- Variabilidad clínica: desde quistes múltiples pequeños (sample_01) hasta acumulaciones masivas (sample_10), pasando por casos casi asintomáticos (sample_25)

### Entregables requeridos
1. Metodología baseline (25%)
2. Mejoras propuestas y aplicadas (25%)
3. Experimentos comparativos con métricas (25%)
4. Informe PDF en formato IEEE (25%)
5. Modelos entrenados (.pth)
6. Defensa oral

---

## Estructura del notebook (`VCA-p2.ipynb`)

### Sección 0 — Setup
- El notebook se ejecuta en **Google Colab** — mantener el montaje de Google Drive y la variable `route` apuntando al directorio del proyecto en Drive
- Todos los imports necesarios
- Configuración de DEVICE (cuda si disponible, si no cpu)
- Semilla global para reproducibilidad

### Sección 1 — Análisis exploratorio del dataset
Análisis estadístico de las imágenes para justificar decisiones de diseño:

**Variables calculadas por imagen:**
- `brightness` — media de intensidad de píxeles (imagen en escala de grises)
- `contrast` — desviación estándar global de intensidad
- `dynamic_range` — diferencia entre percentil 95 y percentil 5 (robusto al fondo negro)
- `fluid_ratio` — porcentaje de píxeles blancos en la máscara (cuantifica el desbalanceo)

**Uso de cada variable:**
- `brightness` + `contrast` → derivan parámetros de `ColorJitter` para augmentation (`factor = std/mean`, igual que P1)
- `dynamic_range` → justifica el uso de CLAHE (imágenes con bajo rango dinámico se benefician)
- `fluid_ratio` → cuantifica el desbalanceo real imagen a imagen, justifica Dice Loss, y verifica que el split train/val/test esté balanceado en severidad de patología

**Plots:**
- Histogramas de cada variable
- Scatter brillo vs contraste
- Distribución de fluid_ratio (desbalanceo por imagen)
- Derivación explícita de parámetros de ColorJitter a partir de los estadísticos

### Sección 2 — Dataset y split
- Reutiliza `OCTDataset` del notebook original
- Split fijo con semilla: **35 train / 5 val / 10 test**
- Los índices del split se fijan antes de cualquier experimento y no cambian

### Sección 3 — Baseline + funciones auxiliares reutilizables
**Todas las funciones auxiliares se definen aquí una sola vez y se reutilizan en todos los experimentos.**

#### Funciones auxiliares de entrenamiento:

**`train_epoch(model, loader, criterion, optimizer, device)`**
- Pone el modelo en `model.train()`
- Itera sobre el loader, hace forward + backward + optimizer step
- Acumula TP, FP, FN, TN a nivel de **píxel** (no de imagen)
- Usa `get_segmentation_masks()` para binarizar predicciones
- Devuelve: loss media, IoU, Dice, precision, recall, accuracy

**`eval_epoch(model, loader, criterion, device)`**
- Pone el modelo en `model.eval()` con `torch.no_grad()`
- Misma lógica de acumulación de métricas que `train_epoch`
- Devuelve: loss media, IoU, Dice, precision, recall, accuracy

**`compute_metrics(tp, fp, fn, tn)`**
- Calcula todas las métricas a partir de TP/FP/FN/TN acumulados
- IoU = TP / (TP + FP + FN)
- Dice = 2·TP / (2·TP + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Accuracy = (TP + TN) / (TP + FP + FN + TN)
- Añade epsilon (1e-8) para evitar división por cero

**`train_model(model, train_loader, val_loader, criterion, max_epochs, early_stop, lr, checkpoint_path, device)`**
- Bucle completo de entrenamiento
- Optimizer: Adam
- Scheduler: `ReduceLROnPlateau(mode='min', patience=5, factor=0.5)`
- Guarda el mejor modelo según **val loss** con `torch.save(model.state_dict(), checkpoint_path)`
- Early stopping si val loss no mejora en `early_stop` épocas consecutivas
- Al finalizar carga el mejor checkpoint con `model.load_state_dict(torch.load(...))`
- Devuelve historial de métricas por época (dict con listas)

**`evaluate_model(model, loader, device, experiment_name)`**
- Evaluación final en test sin criterion
- Calcula todas las métricas sobre el conjunto de test
- Imprime resumen de métricas
- Muestra visualizaciones con `show_result()` para varias imágenes de test

**`plot_training_history(history, experiment_name)`**
- Curvas de loss train/val por época
- Curvas de IoU y Dice train/val por época
- Título con nombre del experimento

#### Entrenamiento del baseline:
- Modelo: `UNet(input_channels=1, n_class=1)` (existente)
- Loss: `BCEWithLogitsLoss`
- Sin augmentation (usa el resize por defecto de `OCTDataset`)
- Checkpoint guardado en `models/baseline.pth`

### Sección 4 — Exp 1: Dice Loss
**Motivación:** `BCEWithLogitsLoss` trata todos los píxeles igual. Con desbalanceo severo (2-10% de píxeles son fluido) la red aprende a predecir todo negro y obtiene alta accuracy pero IoU/Dice nulos. Dice Loss optimiza directamente el coeficiente Dice, robusto al desbalanceo.

**Cambios respecto al baseline:**
- Nueva función `dice_loss(pred, target, smooth=1.0)` — aplica sigmoid internamente
- Se pasa `dice_loss` como `criterion` a `train_model()`
- Todo lo demás igual (misma UNet, mismo split, sin augmentation)
- Checkpoint: `models/exp1_dice.pth`

### Sección 5 — Exp 2: CLAHE
**Motivación:** CLAHE (Contrast Limited Adaptive Histogram Equalization) es el estándar en preprocesado de imágenes OCT. Mejora el contraste local sin saturar, haciendo los bordes del fluido más visibles. El rango dinámico bajo detectado en el análisis exploratorio justifica su uso.

**Implementación:**
- Análisis visual previo: comparación de clipLimit 1.0 / 2.0 / 4.0 sobre 3-4 imágenes representativas
- Se elige el clipLimit que mejor resalta bordes de fluido sin introducir ruido
- CLAHE se aplica en el `__getitem__` de `OCTDataset` como preprocesado previo al transform
- Se aplica solo a la imagen, no a la máscara
- Acumulativo sobre Exp 1 (mantiene Dice Loss)
- Checkpoint: `models/exp2_clahe.pth`

### Sección 6 — Exp 3: Data Augmentation
**Motivación:** Con 35 imágenes de entrenamiento el modelo sobreajusta sin augmentation. El augmentation amplía artificialmente el dataset preservando la validez clínica.

**Parámetros derivados del análisis estadístico (Sección 1):**
- `brightness_factor = b_std / b_mean`
- `contrast_factor = c_std / c_mean`

**Transforms aplicados:**
- `RandomHorizontalFlip(p=0.5)` — la fóvea puede estar a izquierda o derecha
- **NO** vertical flip — invertiría la retina anatómicamente
- `RandomRotation(degrees=10)` — pequeñas variaciones de posicionamiento del paciente
- `ColorJitter(brightness=brightness_factor, contrast=contrast_factor)` — variaciones entre escáneres
- **GaussianBlur descartado** — en OCT la nitidez baja indica mala calidad de adquisición, no variación clínica relevante. Además el fluido ya tiene bordes sutiles y el blur puede dificultar su detección.
- El seed sincronizado de `OCTDataset` garantiza que imagen y máscara reciben la misma transformación geométrica
- ColorJitter se aplica solo a la imagen, no a la máscara

**Acumulativo sobre Exp 2 (mantiene Dice Loss + CLAHE)**
- Checkpoint: `models/exp3_augmentation.pth`

### Sección 7 — Exp 4: Encoder preentrenado (Transfer Learning)
**Motivación:** Con 35 imágenes, entrenar un encoder desde cero es un handicap. Un encoder preentrenado en ImageNet ya detecta bordes, texturas y estructuras relevantes para delimitar el fluido.

**Implementación:**
- `segmentation_models_pytorch.Unet(encoder_name='resnet18', encoder_weights='imagenet', in_channels=1, classes=1)`
- ResNet18 elegido por ser el más ligero y evitar sobreajuste con dataset pequeño
- La imagen de 1 canal es compatible porque smp adapta la primera capa del encoder
- Acumulativo sobre Exp 3 (mantiene Dice Loss + CLAHE + augmentation)
- Checkpoint: `models/exp4_pretrained.pth`

### Sección 8 — Tabla comparativa
Tabla con todas las métricas de test de todos los experimentos:

| Experimento | IoU | Dice | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Baseline (BCE, sin aug) | | | | | |
| Exp 1: + Dice Loss | | | | | |
| Exp 2: + CLAHE | | | | | |
| Exp 3: + Augmentation | | | | | |
| Exp 4: + Encoder preentrenado | | | | | |

---

## Código de partida del profesor (no duplicar)

### `show(image, mask, title=None)`
Muestra imagen y máscara lado a lado en escala de grises. Usada para visualización exploratoria.

### `show_result(orig, gt, prediction, title=None)`
Muestra 4 paneles: imagen original, máscara ground truth, predicción de la red, y overlap (orig * prediction). Usada en `evaluate_model()`.

### `get_segmentation_masks(outputs, threshold=0.5)`
Aplica sigmoid a los logits crudos de la red y binariza con un umbral. Devuelve máscara binaria flotante (0.0 o 1.0).
- `probs = torch.sigmoid(outputs)`
- `masks = (probs > threshold) * 1.0`
- El threshold por defecto es 0.5 pero puede necesitar ajuste (el enunciado lo indica explícitamente)

### `double_conv(in_channels, out_channels)`
Bloque de dos convoluciones 3×3 con ReLU entre ellas. Mismo campo receptivo que una conv 5×5 pero con menos parámetros y más no-linealidades. Base del encoder y decoder de UNet.

### `UNet(input_channels, n_class)`
Arquitectura encoder-decoder con skip connections:
- **Encoder**: 4 niveles — 64, 128, 256, 512 canales. MaxPool2d(2) entre niveles.
- **Bottleneck**: `dconv_down4` (512 canales) — espacio latente, mayor campo receptivo, sin pooling posterior.
- **Decoder**: 3 niveles con Upsample bilineal + concatenación de skip connection + double_conv.
  - `dconv_up3`: recibe 512 (decoder) + 256 (skip de conv3) = 768 canales → 256
  - `dconv_up2`: recibe 256 (decoder) + 128 (skip de conv2) = 384 canales → 128
  - `dconv_up1`: recibe 128 (decoder) + 64 (skip de conv1) = 192 canales → 64
- **Salida**: conv 1×1 → `n_class` canales (logits crudos, sin sigmoid)
- `dim=1` en `torch.cat` concatena en la dimensión de canales [B, C, H, W]
- `Upsample(scale_factor=2, mode='bilinear', align_corners=True)` — bilinear para contornos suaves, align_corners=True para consistencia geométrica en los 3 upsamples encadenados

### `OCTDataset(image_path, mask_path, rsize=(416,624), transform=None)`
Dataset PyTorch que carga imagen+máscara:
- Carga todos los `.jpg` del directorio con `glob`
- Asume que imagen y máscara tienen el mismo nombre en directorios distintos
- Aplica `cv2.threshold(mask, 100, 255, THRESH_BINARY)` para forzar binarización estricta (compensar artefactos de compresión JPEG)
- Si `transform` es None: aplica resize + ToTensor por defecto
- Si `transform` no es None: usa un seed sincronizado (`np.random.randint` → `random.seed` + `torch.manual_seed`) para que imagen y máscara reciban exactamente la misma transformación geométrica aleatoria. ColorJitter y GaussianBlur deben aplicarse solo a la imagen fuera de este mecanismo.

---

## Decisiones de diseño clave

- **Flip vertical prohibido**: invertiría la retina anatómicamente
- **Flip horizontal permitido**: la fóvea puede estar a izquierda o derecha
- **Métricas principales**: IoU y Dice (accuracy es engañosa por el desbalanceo severo)
- **Modelos guardados en**: `p2/OCT-dataset/models/`
- **Un modelo por experimento**: `baseline.pth`, `exp1_dice.pth`, `exp2_clahe.pth`, `exp3_augmentation.pth`, `exp4_pretrained.pth`
- **Semilla fija** para el split train/val/test — no cambia entre experimentos
- **Early stopping** basado en val loss para todos los experimentos
