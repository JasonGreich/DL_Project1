import torch
import torch.nn as nn
from .layers import GConv, GDownSamplingBlock, GUpsamplingBlock
from .attention import ContextualAttention


class CoarseGenerator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = GConv(cnum_in, cnum // 2, 5, 1, padding=2)


        # Downsample Conv
        self.down_block1 = GDownSamplingBlock(cnum // 2, cnum)
        self.down_block2 = GDownSamplingBlock(cnum, 2*cnum)

        # Bottleneck
        self.conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
        self.conv_bn3 = GConv(2 * cnum, 2 * cnum, 3, rate=4, padding=4)
        self.conv_bn4 = GConv(2 * cnum, 2 * cnum, 3, rate=8, padding=8)
        self.conv_bn5 = GConv(2 * cnum, 2 * cnum, 3, rate=16, padding=16)
        self.conv_bn6 = GConv(2 * cnum, 2 * cnum, 3)
        self.conv_bn7 = GConv(2 * cnum, 2 * cnum, 3)


        # Upsample DeConv

        self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

        # RGB
        self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)

        # down
        x = self.down_block1(x)
        x = self.down_block2(x)

        # bottleneck
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = self.conv_bn5(x)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)

        # Up
        x = self.up_block1(x)
        x = self.up_block2(x)

        # RGB
        x = self.conv_to_rgb(x)
        x = self.tanh(x)

        return x


class FineGenerator(nn.Module):
    def __init__(self, cnum, return_flow=False):
        super().__init__()
        self.conv_conv1 = GConv(3, cnum //2, 5, 1, padding=2)
        # down
        self.conv_down_block1 = GDownSamplingBlock(cnum //2, cnum, cnum_hidden=cnum//2)
        self.conv_down_block2 = GDownSamplingBlock(cnum, 2*cnum, cnum_hidden=cnum)

        # bottleneck
        self.conv_conv_bn1 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv_conv_bn2 = GConv(2 * cnum, 2 * cnum, 3, rate=2, padding=2)
        self.conv_conv_bn3 = GConv(2 * cnum, 2 * cnum, 3, rate=4, padding=4)
        self.conv_conv_bn4 = GConv(2 * cnum, 2 * cnum, 3, rate=8, padding=8)
        self.conv_conv_bn5 = GConv(2 * cnum, 2 * cnum, 3, rate=16, padding=16)


        # Contextual Attention

        self.ca_conv1 = GConv(3, cnum//2, 5, 1, padding=2)
        self.ca_down_block1 = GDownSamplingBlock(cnum//2, cnum, cnum_hidden=cnum//2)
        self.ca_down_block2 = GDownSamplingBlock(cnum, 2*cnum)

        # bottleneck

        self.ca_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1, activation=nn.ReLU())

        # body ca
        self.contextual_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True, device_ids=None, return_flow=return_flow, n_down=2)
        self.ca_conv_bn4 = GConv(2*cnum, 2*cnum, 3, 1)
        self.ca_conv_bn5 = GConv(2 * cnum, 2 * cnum, 3, 1)

        self.ca_conv_bn6 = GConv(4*cnum, 2*cnum, 3, 1)
        self.ca_conv_bn7 = GConv(2 * cnum, 2 * cnum, 3, 1)

        # upsample
        self.up_block1 = GUpsamplingBlock(2 * cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)


        # RGB
        self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        xnow = x
        x = self.conv_conv1(xnow)
        # down
        x = self.conv_down_block1(x)
        x = self.conv_down_block2(x)

        #bottleneck
        x = self.conv_conv_bn1(x)
        x = self.conv_conv_bn2(x)
        x = self.conv_conv_bn3(x)
        x = self.conv_conv_bn4(x)
        x = self.conv_conv_bn5(x)
        x_hallu = x

        x = self.ca_conv1(xnow)
        # x = self.ca_conv_bn1(xnow)

        #down
        x = self.ca_down_block1(x)
        x = self.ca_down_block2(x)

        # bottleneck
        x = self.ca_conv_bn1(x)
        x, offset_flow = self.contextual_attention(x,x,  mask)
        x = self.ca_conv_bn4(x)
        x = self.ca_conv_bn5(x)

        pm = x

        #concatenate
        x = torch.cat([x_hallu, pm], dim=1)

        x = self.ca_conv_bn6(x)
        x = self.ca_conv_bn7(x)

        # Upsample
        x = self.up_block1(x)
        x = self.up_block2(x)

        # RGB

        x = self.conv_to_rgb(x)
        x = self.tanh(x)

        return x, offset_flow



def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out



class Generator(nn.Module):
    def __init__(self, cnum_in=5, cnum=48, return_flow=False, checkpoint=None):
        super().__init__()
        self.stage1 = CoarseGenerator(cnum_in, cnum)
        self.stage2 = FineGenerator(cnum, return_flow)
        self.return_flow = return_flow

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
            state_dict = ckpt['G'] if 'G' in ckpt else ckpt

            fixed_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                # rename keys from old model to your current names
                new_k = new_k.replace('stage2.conv_bn6', 'stage2.ca_conv_bn6')
                new_k = new_k.replace('stage2.conv_bn7', 'stage2.ca_conv_bn7')
                fixed_state_dict[new_k] = v

            # you can keep strict=True now
            self.load_state_dict(fixed_state_dict, strict=True)

        self.eval()



    def forward(self, x, mask):
        xin = x
        # get coarse result
        x_stage1 = self.stage1(x)
        # inpaint input with coarse result
        x = x_stage1*mask + xin[:, 0:3, :, :]*(1.-mask)
        # get refined result
        x_stage2, offset_flow = self.stage2(x, mask)

        if self.return_flow:
            return x_stage1, x_stage2, offset_flow

        return x_stage1, x_stage2

    @torch.inference_mode()
    def infer(self,
              image,
              mask,
              return_vals=['inpainted', 'stage1'],
              device='cuda'):
        """
        Args:
            image:
            mask:
            return_vals: inpainted, stage1, stage2, flow
        Returns:
        """

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        image = (image*2 - 1.)  # map image values to [-1, 1] range
        # 1.: masked 0.: unmasked
        mask = (mask > 0.).to(dtype=torch.float32)

        image_masked = image * (1.-mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # sketch channel
        x = torch.cat([image_masked, ones_x, ones_x*mask],
                      dim=1)  # concatenate channels

        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)

        image_compl = image * (1.-mask) + x_stage2 * mask

        output = []
        for return_val in return_vals:
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')

        return output
