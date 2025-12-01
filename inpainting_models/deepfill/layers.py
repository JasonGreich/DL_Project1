import torch
import torch.nn as nn
import torch.nn.functional as F

class GConv(nn.Module):
    def __init__(self, num_in, num_out, ksize, stride=1,padding='auto', rate=1, activation=nn.ELU(), bias=True):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        num_out_final = num_out if num_out == 3 or activation is None else 2 * num_out
        self.activation = activation
        self.ksize = ksize
        padding = rate * (ksize - 1) // 2 if padding == 'auto' else padding
        self.padding = padding
        self.stride = stride


        self.conv = nn.Conv2d(self.num_in, num_out_final, kernel_size=ksize, padding=padding, stride=stride, bias=bias, dilation=rate)

    def forward(self, x):
        x = self.conv(x)
        if self.num_out == 3 or self.activation is None:
            return x
        x, y = torch.split(x, self.num_out, dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x



class GDeConv(nn.Module):
    def __init__(self, cnum_in, cnum_out, padding=1):
        super().__init__()
        self.conv = GConv(cnum_in, cnum_out,3, stride=1, padding=padding)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest", recompute_scale_factor=False)
        x = self.conv(x)
        return x

class GDownSamplingBlock(nn.Module):
    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden is None else cnum_hidden

        self.conv1_downsample = GConv(cnum_in, cnum_hidden, 3, stride=2)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, stride=1)


    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x


class GUpsamplingBlock(nn.Module):
    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden is None else cnum_hidden

        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, stride=1)


    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        return x


class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in,
                 cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        weight_orig = self.weight.flatten(1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x



class DConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out, ksize=5, stride=2, padding='auto'):
        super().__init__()
        padding = (ksize-1)//2 if padding == 'auto' else padding
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)
        #self.conv_sn = spectral_norm(nn.Conv2d(cnum_in, cnum_out, ksize, stride, padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x


