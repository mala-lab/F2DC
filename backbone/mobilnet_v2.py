import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.gumbel_sigmoid import GumbelSigmoid


class DFD(torch.nn.Module):
    def __init__(self, size, num_channel=64, tau=0.1):
        super(DFD, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat, is_eval=False):
        rob_map = self.net(feat)
        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)

        r_feat = feat * mask
        nr_feat = feat * (1 - mask)

        return r_feat, nr_feat, mask


class DFC(nn.Module):
    def __init__(self, size, num_channel=64):
        super(DFC, self).__init__()
        C, H, W = size
        self.net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, nr_feat, mask):
        rec_units = self.net(nr_feat)
        rec_units = rec_units * (1 - mask)
        rec_feat = nr_feat + rec_units

        return rec_feat


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, tau=0.1, image_size=(32, 32), name='f2dc'):
        super(MobileNetV2, self).__init__()
        self.name = name
        self.in_planes = 64
        self.tau = tau
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

        self.encoder = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 512)
        )

        self.dfd_module = DFD(size=(512, int(self.image_size[0] / 8), int(self.image_size[1] / 8)), tau=self.tau)
        self.dfc_module = DFC(size=(512, int(self.image_size[0] / 8), int(self.image_size[1] / 8)))
        self.aux = nn.Sequential(nn.Linear(512, num_classes))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)


    def forward(self, x, is_eval=False):
        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # print(out.shape)
        r_feat, nr_feat, mask = self.dfd_module(out, is_eval=is_eval)
        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)
        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        rec_feat = self.dfc_module(nr_feat, mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)
        out = r_feat + rec_feat
        # out = F.avg_pool2d(out, 4)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)
        return out, feat, r_outputs, nr_outputs, rec_outputs

    def encoders(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.encoder(out)
        return out

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

    def fc(self, x):
        out = self.linear(x)
        return out


def mobile_dc(num_classes=7, gum_tau=0.1):
    return MobileNetV2(num_classes=num_classes, tau=gum_tau, image_size=(128, 128))

def mobile_dc_office(num_classes=10, gum_tau=0.1):
    return MobileNetV2(num_classes=num_classes, tau=gum_tau, image_size=(32, 32))

def mobile_dc_digits(num_classes=10, gum_tau=0.1):
    return MobileNetV2(num_classes=num_classes, tau=gum_tau, image_size=(32, 32))


# if __name__=="__main__":
#     mnet = MobileNetV2(num_classes=10, tau=0.1, image_size=(128, 128))
#     aa = torch.rand(1,3,128,128)
#     out, feat, r_outputs, nr_outputs, rec_outputs = mnet(aa)