import copy
import torch
from torch import nn
from model.module.bnneck import BNNeck, BNNeck3
from torch.autograd import Variable
from model.module.cbam import CBAM, ConvinteractLoc
from model.module.OSBSR import get_decoder
from model.resnet import resnet50, Bottleneck_IBN
import logging


class MaxAvgPlus(nn.Module):
    def __init__(self, in_fea=512, use_relu=False, num_stripes=1, out_fea=None):
        super(MaxAvgPlus, self).__init__()
        if out_fea is None:
            out_fea = in_fea
        self.avgpool = nn.AdaptiveAvgPool2d((num_stripes, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((num_stripes, 1))
        self.conv = nn.Conv2d(in_fea*2, out_fea, 1, bias=False)
        if use_relu:
            self.relu = nn.ReLU(True)
        else:
            self.relu = None

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x_out = self.conv(torch.cat([x_avg, x_max], dim=1))
        if self.relu:
            x_out = self.relu(x_out)
        return x_out


class Res50_D_MDIFF(nn.Module):
    def __init__(self, num_class=751, FDR=False, cfg=None, **kwargs):
        super(Res50_D_MDIFF, self).__init__()
        self.neck_feat = 'after'
        if FDR:
            p1_p2_dim = 256
        else:
            p1_p2_dim = 512
        logger = logging.getLogger("MDIFF_CRReID")
        logger.info(f'Making Res50_D_MDIFF model with FDR={FDR}, glo_feat_dim/p1_p2_dim={p1_p2_dim}')
        base = resnet50(last_stride=1)
        self.backone = nn.Sequential(*list(base.children())[:6])
        base_cp = copy.deepcopy(base)
        self.backtwo = nn.Sequential(*list(base_cp.children())[:5], CBAM(256, residual=True),
                                     *list(base_cp.children())[5], CBAM(512, residual=True))
        self.sr_branch = get_decoder()

        self.partial2_branch = nn.Sequential(copy.deepcopy(base.layer3), copy.deepcopy(base.layer4))
        self.batch_att_block = ConvinteractLoc(2048, Bottleneck_IBN)  # local scale features interaction
        self.conv_all_fuss = nn.Conv2d(2560, 512 * 2, 1, bias=False)  # local and global feature interaction

        self.fcneck_par1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.fcneck_par2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)

        self.map_par1 = MaxAvgPlus(in_fea=512, use_relu=False, out_fea=p1_p2_dim)
        self.map_all = MaxAvgPlus(in_fea=512 * 2, use_relu=False, out_fea=1024)
        self.map_par2 = MaxAvgPlus(in_fea=512, use_relu=False, out_fea=p1_p2_dim)
        self.map_par2_att = MaxAvgPlus(in_fea=512 * 3, use_relu=False, out_fea=512)

        self.BNNeck_p1 = BNNeck(p1_p2_dim, num_class, return_f=True, drop=0.0)
        self.BNNeck_p2 = BNNeck(p1_p2_dim, num_class, return_f=True, drop=0.0)
        self.BNNeck_p2_spat = BNNeck(512, num_class, return_f=True, drop=0.0)
        self.BNNeck_fuss_all = BNNeck(1024, num_class, return_f=True, drop=0.0)

    def forward(self, x, img_type='lr', **kwargs):
        if img_type == 'hr':
            sr = x
            x = self.backone(x)
        else:
            x = self.backtwo(x)
            sr = self.sr_branch(x)

        par1 = self.partial2_branch(x)
        par2_att, par2 = self.batch_att_block(par1)
        par2_att_for_fus = torch.cat([par2_att, par2_att, par2_att], dim=2)
        par1 = self.fcneck_par1(par1)
        par2 = self.fcneck_par2(par2)
        all_fuss_bp = self.conv_all_fuss(torch.cat([par1, par2, par2_att_for_fus], dim=1))

        all_fuss = self.map_all(all_fuss_bp)
        p1 = self.map_par1(par1)
        p2 = self.map_par2(par2)
        p2_att = self.map_par2_att(par2_att)

        f_p1 = self.BNNeck_p1(p1)
        f_p2 = self.BNNeck_p2(p2)
        f_p2_att = self.BNNeck_p2_spat(p2_att)
        f_all_fuss = self.BNNeck_fuss_all(all_fuss)

        fea = [f_p1[-1], f_p2[-1], f_p2_att[-1]]

        if not self.training:
            if self.neck_feat == 'after':
                return torch.cat([f_all_fuss[0], f_p1[0], f_p2[0], f_p2_att[0]], dim=1), sr
            else:
                return torch.cat([f_all_fuss[-1], f_p1[-1], f_p2[-1], f_p2_att[-1]], dim=1), None
        else:
            return [f_all_fuss[1], f_p1[1], f_p2[1], f_p2_att[1]], fea, \
                    [[f_p1[-1], f_p2[-1], f_p2_att[-1], f_all_fuss[-1]], sr, x]

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        exclude_keys = []
        for i in param_dict:
            # if 'map111' in i:
            if ('map' in i) or ('BNNeck' in i) or ('fcneck' in i):
                # print(i)
                exclude_keys.append(i)
            else:
                try:
                    self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                except:
                    exclude_keys.append(i)
                    continue
        print('Loading pretrained model from {}'.format(trained_path))
        return exclude_keys


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.

    print()

