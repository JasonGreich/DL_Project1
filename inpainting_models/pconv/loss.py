import torch
from torch import nn


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def dialation_holes(hole_mask):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)

    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = output_mask != 0
    return updated_holes.float()


def total_variation_loss(image, mask, method):
    hole_mask = 1 - mask
    dilated_holes = dialation_holes(hole_mask)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]

    if method == 'sum':
        loss = torch.sum(torch.abs(colomns_in_Pset * (image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
               torch.sum(torch.abs(rows_in_Pset * (image[:, :, :1, :] - image[:, :, -1:, :])))
    else:
        loss = torch.mean(torch.abs(colomns_in_Pset * (image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
               torch.mean(torch.abs(rows_in_Pset * (image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor, tv_loss='mean'):
        super().__init__()
        self.tv_loss = tv_loss
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        comp = mask * input + (1 - mask) * output

        tv_loss = total_variation_loss(comp, mask, self.tv_loss)
        hole_loss = self.l1((1 - mask) * output, (1 - mask) * gt)
        valid_loss = self.l1(mask * output, mask * gt)

        feats_out = self.extractor(output)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        perc_loss = 0.0
        style_loss = 0.0

        for i in range(3):
            perc_loss += self.l1(feats_out[i], feats_gt[i])
            perc_loss += self.l1(feats_comp[i], feats_gt[i])
            style_loss += self.l1(gram_matrix(feats_out[i]), gram_matrix(feats_gt[i]))
            style_loss += self.l1(gram_matrix(feats_comp[i]), gram_matrix(feats_gt[i]))

        return {'valid': valid_loss, 'hole': hole_loss, 'perc': perc_loss, 'style': style_loss, 'tv': tv_loss}