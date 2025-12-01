import torch
from torch import nn
from .layers import PConvLayer

class PConvUNet(nn.Module):
    def __init__(self, finetune=False, in_ch=3, layer_size=6):
        super().__init__()
        self.freeze_enc_bn = True if finetune else False
        self.layer_size = layer_size

        self.enc_1 = PConvLayer(in_ch, 64, 'down-7', bn=False)
        self.enc_2 = PConvLayer(64, 128, 'down-5', bn=False)
        self.enc_3 = PConvLayer(128, 256, 'down-5', bn=False)
        self.enc_4 = PConvLayer(256, 512, 'down-3', bn=False)
        self.enc_5 = PConvLayer(512, 512, 'down-3', bn=False)
        self.enc_6 = PConvLayer(512, 512, 'down-3', bn=False)
        self.enc_7 = PConvLayer(512, 512, 'down-3', bn=False)
        self.enc_8 = PConvLayer(512, 512, 'down-3', bn=False)

        self.dec_8 = PConvLayer(512 + 512, 512, dec=True, active='leaky', bn=False)
        self.dec_7 = PConvLayer(512 + 512, 512, dec=True, active='leaky', bn=False)
        self.dec_6 = PConvLayer(512 + 512, 512, dec=True, active='leaky', bn=False)
        self.dec_5 = PConvLayer(512 + 512, 512, dec=True, active='leaky', bn=False)
        self.dec_4 = PConvLayer(512 + 256, 256, dec=True, active='leaky', bn=False)
        self.dec_3 = PConvLayer(256 + 128, 128, dec=True, active='leaky', bn=False)
        self.dec_2 = PConvLayer(128 + 64, 64, dec=True, active='leaky', bn=False)
        self.dec_1 = PConvLayer(64 + 3, 3, dec=True, bn=False, active=None, conv_bias=True)

    def forward(self, img, mask):
        enc_f, enc_m = [img], [mask]

        for layer_num in range(1, self.layer_size + 1):
            if layer_num == 1:
                feature, update_mask = getattr(self, f'enc_{layer_num}')(img, mask)
            else:
                enc_f.append(feature)
                enc_m.append(update_mask)
                feature, update_mask = getattr(self, f'enc_{layer_num}')(feature, update_mask)

        assert len(enc_f) == self.layer_size

        for layer_num in reversed(range(1, self.layer_size + 1)):
            feature, update_mask = getattr(self, f'dec_{layer_num}')(feature, update_mask, enc_f.pop(), enc_m.pop())

        return feature, mask

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


# Testing
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PConvUNet(in_ch=3, layer_size=6).to(device)
    input_image = torch.rand(4, 3, 256, 256).to(device)
    input_mask = torch.zeros_like(input_image).to(device)
    input_mask[:, :, 64:192, 64:192] = 1
    model.eval()
    with torch.no_grad():
        output_image, output_mask = model(input_image, input_mask)

    shapes = [
        output_image.shape,
        output_mask.shape
    ]

    print(shapes[0]) if all(s == shapes[0] for s in shapes) else print(shapes)


if __name__ == '__main__':
    main()