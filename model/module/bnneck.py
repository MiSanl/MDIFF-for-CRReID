import copy

from torch import nn

class BNNeck(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False, drop=0.0):
        super(BNNeck, self).__init__()
        self.return_f = return_f
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)
        if drop > 0.0:
            self.dropout = nn.Dropout(drop)
        else:
            self.dropout = None

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)
        if self.dropout is not None:
            after_neck = self.dropout(after_neck)
        else:
            after_neck = after_neck

        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck  # before_neck 计算triplet
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck3(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck3, self).__init__()
        self.return_f = return_f
        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False)

        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        # before_neck = x.squeeze(dim=3).squeeze(dim=2)
        # after_neck = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)
        if self.return_f:
            score = self.classifier(after_neck)
            return after_neck, score, before_neck  # before_neck 计算triplet
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck4(nn.Module):
    def __init__(self, input_dim, class_num, feat_dim, return_f=False):
        super(BNNeck4, self).__init__()
        self.return_f = return_f
        self.reduction = nn.Conv2d(
            input_dim, feat_dim, 1, bias=False)

        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, class_num, bias=False)
        self.proj = nn.Linear(input_dim, feat_dim, bias=False)
        self.bn.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)
        self.proj.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.reduction(x)
        # before_neck = x.squeeze(dim=3).squeeze(dim=2)
        # after_neck = self.bn(x).squeeze(dim=3).squeeze(dim=2)
        before_neck = x.view(x.size(0), x.size(1))
        after_neck = self.bn(before_neck)
        if self.return_f:
            score = self.classifier(after_neck)
            after_proj = self.proj(before_neck)
            return after_neck, score, after_proj, before_neck  # before_neck 计算triplet
        else:
            x = self.classifier(x)
            return x

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck_3par(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False, feat_dim=None):
        super(BNNeck_3par, self).__init__()
        self.return_f = return_f
        if feat_dim is not None:
            mid_dim = feat_dim
        else:
            mid_dim = input_dim
        bn = nn.BatchNorm1d(mid_dim)
        bn.bias.requires_grad_(False)
        bn.apply(self.weights_init_kaiming)
        self.bn1 = copy.deepcopy(bn)
        self.bn2 = copy.deepcopy(bn)
        self.bn3 = copy.deepcopy(bn)
        classifier = nn.Linear(mid_dim, class_num, bias=False)
        classifier.apply(self.weights_init_classifier)
        self.fc1 = copy.deepcopy(classifier)
        self.fc2 = copy.deepcopy(classifier)
        self.fc3 = copy.deepcopy(classifier)
        if feat_dim is not None:
            self.reduction = nn.Conv2d(input_dim, feat_dim, 1, bias=False)
        else:
            self.reduction = None

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        x1 = x[:, :, 0, :].unsqueeze(-1)
        x2 = x[:, :, 1, :].unsqueeze(-1)
        x3 = x[:, :, 2, :].unsqueeze(-1)
        before_1 = x1.view(x1.size(0), x1.size(1))
        before_2 = x2.view(x2.size(0), x2.size(1))
        before_3 = x3.view(x3.size(0), x3.size(1))
        after_1 = self.bn1(before_1)
        after_2 = self.bn2(before_2)
        after_3 = self.bn3(before_3)
        score_1 = self.fc1(after_1)
        score_2 = self.fc2(after_2)
        score_3 = self.fc3(after_3)
        return [after_1, after_2, after_3], [score_1, score_2, score_3], [before_1, before_2, before_3]  # before_neck 计算triplet

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BNNeck_2par(nn.Module):
    def __init__(self, input_dim, class_num, return_f=False):
        super(BNNeck_2par, self).__init__()
        self.return_f = return_f
        bn = nn.BatchNorm1d(input_dim)
        bn.bias.requires_grad_(False)
        bn.apply(self.weights_init_kaiming)
        self.bn1 = copy.deepcopy(bn)
        self.bn2 = copy.deepcopy(bn)
        classifier = nn.Linear(input_dim, class_num, bias=False)
        classifier.apply(self.weights_init_classifier)
        self.fc1 = copy.deepcopy(classifier)
        self.fc2 = copy.deepcopy(classifier)

    def forward(self, x):
        x1 = x[:, :, 0, :].unsqueeze(-1)
        x2 = x[:, :, 1, :].unsqueeze(-1)
        before_1 = x1.view(x1.size(0), x1.size(1))
        before_2 = x2.view(x2.size(0), x2.size(1))
        after_1 = self.bn1(before_1)
        after_2 = self.bn2(before_2)
        score_1 = self.fc1(after_1)
        score_2 = self.fc2(after_2)
        return [after_1, after_2], [score_1, score_2], [before_1, before_2]  # before_neck 计算triplet

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

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x.squeeze(3).squeeze(2))
        if self.return_f:
            f = x
            x = self.classifier(x)
            return f, x, f
        else:
            x = self.classifier(x)
            return x

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            # For old pytorch, you may use kaiming_normal.
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)


class ProJectF(nn.Module):
    def __init__(self, input_dim):
        super(ProJectF, self).__init__()
        self.classifier = nn.Linear(input_dim, input_dim, bias=False)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        before_neck = x.view(x.size(0), x.size(1))
        score = self.classifier(before_neck)
        return score

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

