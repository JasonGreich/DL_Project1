"""
UNet implementation with ResNet-34 encoder for semantic segmentation.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling block with skip connections"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """
    UNet with ResNet-34 encoder.

    Args:
        num_classes: Number of output classes (183 for COCO-Stuff)
        pretrained: Whether to use pretrained ResNet-34
    """
    def __init__(self, num_classes=183, pretrained=True, freeze_encoder=True):
        super().__init__()

        # Load pretrained ResNet-34
        if pretrained:
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34(weights=None)

        if freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        # Encoder (ResNet-34 layers)
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels, /2

        self.encoder2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )  # 64 channels, /4

        self.encoder3 = resnet.layer2  # 128 channels, /8
        self.encoder4 = resnet.layer3  # 256 channels, /16
        self.encoder5 = resnet.layer4  # 512 channels, /32

        # Decoder (upsampling path)
        self.up1 = Up(512 + 256, 256)  # 512 from encoder5 + 256 from encoder4
        self.up2 = Up(256 + 128, 128)  # 256 from up1 + 128 from encoder3
        self.up3 = Up(128 + 64, 64)    # 128 from up2 + 64 from encoder2
        self.up4 = Up(64 + 64, 64)     # 64 from up3 + 64 from encoder1

        # Final upsampling to original resolution
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)      # [B, 64, H/2, W/2]
        e2 = self.encoder2(e1)     # [B, 64, H/4, W/4]
        e3 = self.encoder3(e2)     # [B, 128, H/8, W/8]
        e4 = self.encoder4(e3)     # [B, 256, H/16, W/16]
        e5 = self.encoder5(e4)     # [B, 512, H/32, W/32]

        # Decoder with skip connections
        d1 = self.up1(e5, e4)      # [B, 256, H/16, W/16]
        d2 = self.up2(d1, e3)      # [B, 128, H/8, W/8]
        d3 = self.up3(d2, e2)      # [B, 64, H/4, W/4]
        d4 = self.up4(d3, e1)      # [B, 64, H/2, W/2]

        # Final upsampling and output
        d5 = self.final_up(d4)     # [B, 64, H, W]
        out = self.out_conv(d5)    # [B, num_classes, H, W]

        return out


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] class indices
        """
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, pred.size(1))
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss"""
    def __init__(self, num_classes=183, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def test_unet():
    """Test UNet model"""
    model = ResNetUNet(num_classes=183, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 183, 512, 512), f"Expected (2, 183, 512, 512), got {y.shape}"

    # Test loss
    target = torch.randint(0, 183, (2, 512, 512))
    loss_fn = SegmentationLoss()
    loss = loss_fn(y, target)
    print(f"Loss: {loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == '__main__':
    test_unet()
