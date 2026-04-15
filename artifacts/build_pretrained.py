import torch.nn as nn
import torchvision.models as models


def build_pretrained():
    model = models.resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
