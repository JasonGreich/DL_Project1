import torch
from torch import nn
import torch.nn.functional as F

class PartialConvolution(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                         padding_mode=padding_mode)

        # Kernel for updating mask
        self.mask_kernel = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.register_buffer('mask_kernel_buffer', self.mask_kernel)

        # For renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] * self.mask_kernel.shape[3]

        # Initialize weights for image convolution
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):
        mask_kernel = self.mask_kernel_buffer

        # Calculate update mask
        update_mask = F.conv2d(mask, mask_kernel, bias=None, stride=self.stride, padding=self.padding,
                               dilation=self.dilation, groups=1)
        mask_ratio = self.sum1 / (update_mask + 1e-8)
        update_mask = torch.clamp(update_mask, 0, 1)
        mask_ratio = mask_ratio * update_mask

        # Apply partial convolution
        conved = F.conv2d(img * mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Apply mask ratio normalization
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = mask_ratio * (conved - bias_view) + bias_view
        else:
            output = conved * mask_ratio

        return output, update_mask


class UpsampleData(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, dec_feature, enc_feature, dec_mask, enc_mask):
        out = torch.cat([self.upsample(dec_feature), enc_feature], dim=1)
        out_mask = torch.cat([self.upsample(dec_mask), enc_mask], dim=1)
        return out, out_mask


class PConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', dec=False, bn=True, active='relu', conv_bias=False):
        super().__init__()

        kernel_map = {'down-7': (7, 2, 3), 'down-5': (5, 2, 2), 'down-3': (3, 2, 1)}
        kernel_size, stride, padding = kernel_map.get(sample, (3, 1, 1))

        self.conv = PartialConvolution(in_ch, out_ch, kernel_size, stride, padding, bias=conv_bias)
        if dec:
            self.upcat = UpsampleData()
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if active == 'relu':
            self.activation = nn.ReLU()
        elif active == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, img, mask, enc_img=None, enc_mask=None):
        if hasattr(self, 'upcat'):
            out, update_mask = self.upcat(img, enc_img, mask, enc_mask)
            out, update_mask = self.conv(out, update_mask)
        else:
            out, update_mask = self.conv(img, mask)

        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)

        return out, update_mask