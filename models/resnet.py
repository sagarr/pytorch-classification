import torchvision
from torch import nn


def get_model(num_classes, name):
    print("=Building ", name)

    model = getattr(torchvision.models, name)(pretrained=True)

    features = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)

    ct = []
    for name, child in model.named_children():
        if "layer1" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)

    return model
