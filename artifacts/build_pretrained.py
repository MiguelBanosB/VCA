import torch.nn as nn
import torchvision.models as models


class _LinearSqueeze(nn.Module):
    """Wrapper que aplica squeeze(1) tras el fc para devolver [B] en lugar de [B, 1].
    Debe coincidir exactamente con la arquitectura usada en el entrenamiento."""
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)  # [B, 1] -> [B]


def build_pretrained():
    model = models.resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    # Debe coincidir con el notebook — claves del state_dict: fc.fc.weight / fc.fc.bias
    model.fc = _LinearSqueeze(model.fc.in_features)

    return model
