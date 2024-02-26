
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.module.Upsamplers as Upsamplers
from model.module.osnet import OSBlock, ConvLayer, osnet_x1_0


class OSBD(nn.Module):
    def __init__(self, num_out_ch=3, upscale=4):
        super(OSBD, self).__init__()

        self.fea_extract = OSBlock(384, 256)
        self.feat_upsampler = nn.Upsample(scale_factor=2, mode='bilinear')  # bicubic  bilinear
        self.GELU = nn.GELU()
        self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=256, num_out_ch=num_out_ch)

    def forward(self, input):

        out_fea = self.fea_extract(input)
        out_fea = self.feat_upsampler(out_fea)
        out_fea = self.GELU(out_fea)
        output_sr = self.upsampler(out_fea)  # 上采样，并降低通道数

        return output_sr


def get_decoder(scale=4, in_ch=512):
    model = OSBD(num_out_ch=3, upscale=scale)
    model.fea_extract = OSBlock(in_ch, 256)
    return model


if __name__ == '__main__':

    print('done!!!')
