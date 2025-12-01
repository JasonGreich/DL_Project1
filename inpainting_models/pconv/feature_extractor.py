import torch
from torch import nn
from torchvision.models import VGG16_Weights
import torchvision.models as models


class VGG16FeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        normalization = Normalization(self.MEAN, self.STD)

        # Feature extractor layers
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # Freeze all VGG layers
        for enc_layer in [self.enc_1, self.enc_2, self.enc_3]:
            for param in enc_layer.parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(self, f'enc_{i + 1}')(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        if self.mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
        return (input - self.mean) / self.std