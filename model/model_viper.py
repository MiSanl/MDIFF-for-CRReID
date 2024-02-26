import copy
import torch
from torch import nn
from model.module.bnneck import BNNeck, BNNeck3
from torch.autograd import Variable
from model.module.cbam import CBAM, ConvSpFus
from model.module.OSBSR import get_decoder
from model.resnet import resnet50, Bottleneck_IBN


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


class res50_d_d_att_map_simple_pcb(nn.Module):
    '''
    self.base_1 = nn.Sequential(*list(self.base.children())[:6])  获取layer3(不包括)之前的模块
    直接子模块是指通过在 nn.Module 的 __init__ 方法中使用 self.xxx = ... 语句定义的属性，而 self.frozen_stages 只是一个普通的变量，不是通过继承 nn.Module 创建的子模块。
    在 ResNet_IBN 类中，self.frozen_stages 用于指定需要冻结的层的阶段数。它是一个标量整数值，并不代表一个模块或子模块。
    '''

    def __init__(self, args, num_class=751, cfg=None, **kwargs):
        super(res50_d_d_att_map_simple_pcb, self).__init__()
        self.cross_tri = cfg.MODEL.CROSS_TRIPLET
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.att_type = cfg.MODEL.ADDED_ATT
        base = resnet50_ibn_a(last_stride=1)
        # base = resnet50(last_stride=1)
        model_path = '/home/bobo/H/zwc/CrossResolution/pretrained/r50_ibn_a.pth'
        base.load_param(model_path)  # 从 ImageNet 初始化参数
        print(f'Loading ImageNet pretrained model...{model_path}')
        self.backone = nn.Sequential(*list(base.children())[:6])  # self.base[:layer3]
        base_cp = copy.deepcopy(base)
        self.backtwo = nn.Sequential(*list(base_cp.children())[:5], CBAM(256, residual=True),
                                     *list(base_cp.children())[5], CBAM(512, residual=True))  # 暂时remove cuhk03数据集
        # self.backtwo = copy.deepcopy(self.backone)
        self.sr_branch = get_OSBD_resnet()  # 暂时remove cuhk03数据集

        self.partial2_branch = nn.Sequential(copy.deepcopy(base.layer3), copy.deepcopy(base.layer4))
        # 0827晚测试，使用convatt效果最好
        self.batch_att_block = ConvSpFus(2048, Bottleneck_IBN)
        self.conv_all_fuss = nn.Conv2d(2560, 512 * 2, 1, bias=False)  # after add fcneck

        print(f'===>>> Attention Type to get par_att is {self.att_type}')

        self.fcneck_par1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)  # add fcneck before BNNeck, replace BNNeck3
        self.fcneck_par2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        # self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        # self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # self.conv_par_fuss = nn.Conv2d(512 * 2, 512, 1, bias=False)  # p1+p2 fusion after add fcneck
        self.map_par1 = MaxAvgPlus(in_fea=512, use_relu=False, out_fea=256)  # after add fcneck
        self.map_all = MaxAvgPlus(in_fea=512 * 2, use_relu=False, out_fea=1024)
        self.map_par2 = MaxAvgPlus(in_fea=512, use_relu=False, out_fea=256)
        self.map_par2_att = MaxAvgPlus(in_fea=512 * 3, use_relu=False, out_fea=512)
        for name, parameter in self.named_parameters():
            if 'map' in name:
                parameter.requires_grad = False

        self.BNNeck_p1 = BNNeck(256, num_class, return_f=True, drop=0.0)
        self.BNNeck_p2 = BNNeck(256, num_class, return_f=True, drop=0.0)  # after add fcneck
        # self.BNNeck_par_fuss = BNNeck(512, num_class, return_f=True)  # after add fcneck
        self.BNNeck_p2_spat = BNNeck(512, num_class, return_f=True, drop=0.0)  # after Bottleneck_IBN + SPAT_block
        self.BNNeck_fuss_all = BNNeck(1024, num_class, return_f=True, drop=0.0)

    def forward(self, x, img_type='lr', **kwargs):
        if img_type == 'hr':
            sr = x
            x = self.backone(x)
        else:
            # sr = x
            x = self.backtwo(x)
            sr = self.sr_branch(x)

        par1 = self.partial2_branch(x)  # added
        par2_att, par2 = self.batch_att_block(par1)  # glo
        par2_att_for_fus = torch.cat([par2_att, par2_att, par2_att], dim=2)
        par1 = self.fcneck_par1(par1)
        par2 = self.fcneck_par2(par2)
        all_fuss_bp = self.conv_all_fuss(torch.cat([par1, par2, par2_att_for_fus], dim=1))

        all_fuss = self.map_all(all_fuss_bp)
        p1 = self.map_par1(par1)  # shape:(batchsize, 512,1,1) 0.524M
        # p2 = self.avg_pooling(par2)  # shape:(batchsize, 512,1,1)
        p2 = self.map_par2(par2)  # shape:(batchsize, 512,1,1)
        # p2_att = self.max_pooling(par2_att)  # (B, 1536, 1, 1)
        p2_att = self.map_par2_att(par2_att)  # (B, 1536, 1, 1)
        # p_fuss = self.conv_par_fuss(torch.cat([p1, p2], dim=1))  # (B, 512, 1, 1) 0.524M

        f_p1 = self.BNNeck_p1(p1)  # after_neck(for id & test), score, before_neck(for tri loss)
        f_p2 = self.BNNeck_p2(p2)
        f_p2_att = self.BNNeck_p2_spat(p2_att)
        # f_p_fuss = self.BNNeck_par_fuss(p_fuss)  # rm-par-fus
        f_all_fuss = self.BNNeck_fuss_all(all_fuss)

        fea = [f_p1[-1], f_p2[-1], f_p2_att[-1]]

        if not self.training:
            if self.neck_feat == 'after':
                return torch.cat([f_all_fuss[0], f_p1[0], f_p2[0], f_p2_att[0]], dim=1), None  # rm
            else:
                return torch.cat([f_all_fuss[-1], f_p1[-1], f_p2[-1], f_p2_att[-1]], dim=1), None
        else:
            if self.cross_tri:
                return [f_all_fuss[1], f_p1[1], f_p2[1], f_p2_att[1]], fea, \
                    [[f_p1[-1], f_p2[-1], f_p2_att[-1], f_all_fuss[-1]], sr, x]  # rm , f_p_fuss[1], [-1]
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
            if ('map' in i) or ('BNNeck' in i) or ('fcneck' in i):
                # print(i)
                exclude_keys.append(i)
            else:
                try:
                    self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                except:
                    continue
        print('Loading pretrained model from {}'.format(trained_path))
        return exclude_keys


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    from config import cfg  # 相当于将defaults.py导入为cfg
    import argparse
    from utils.model_complexity import compute_model_complexity
    from torchinfo import summary

    parser = argparse.ArgumentParser(description='MGN')
    # parser.add_argument('--num_classes', type=int, default=751, help='')
    parser.add_argument('--feats', type=int, default=512)
    parser.add_argument("--activation_map", action='store_true',
                        help='if raise, return feature activation map')

    args = parser.parse_args()
    cfg.MODEL.ADDED_ATT = 'convatt'
    net = res50_d_d_att_map_simple_pcb(args, cfg=cfg)
    # num_params, flops = compute_model_complexity(net, (1,3,384,128))
    # print(summary(net, input_size=(1, 3, 384, 128), verbose=0))
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 384, 128))
    net.eval()
    output, _ = net(input)
    print(f'net output size:{output.shape}')
    print()

