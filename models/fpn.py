import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34

class FPN(nn.Module):
    '''
    - Simple encoder decoder model with by-pass connections.
    - Loss at multiple scales.
    '''
    def __init__(self, num_kpt, base_pretrained=True):
        super(FPN, self).__init__()

        base = resnet34(pretrained=base_pretrained)
        base_channels = [64, 128, 256, 512]

        self.enc = nn.ModuleList([
            nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            ),
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        ])

        self.classifiers = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        C = base_channels[-1]
        for in_channels in reversed(base_channels[:-1]):

            self.decoder.append(nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),

                nn.Conv2d(C, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))

            self.classifiers.append(nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, num_kpt, 1),
                nn.BatchNorm2d(num_kpt),
                nn.Sigmoid()
            ))

            C = in_channels

        self.classifiers.append(nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, num_kpt, 1),
                nn.BatchNorm2d(num_kpt),
                nn.Sigmoid()
        ))

        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        residuals = []
        for e in self.enc:
            x = e(x)
            residuals.append(x)
        residuals.pop(-1)
        residuals.reverse()

        preds = []
        for i, d in enumerate(self.decoder):
            if self.training:
                preds.append(self.classifiers[i](x))

            _, _, h, w = residuals[i].shape
            x = F.interpolate(x, size=(h, w))
            x = d(x)
            x = x + residuals[i]
        preds.append(self.classifiers[-1](x))

        return preds

    def calc_loss(self, preds, hm):
        factor = {}
        loss = {}
        for i, pred in enumerate(preds):
            _, _, h, w = pred.shape
            hm_gt = F.interpolate(hm, (h, w), mode="bilinear", align_corners=True)
            loss['mse_%d'%i] = self.mse_loss(pred, hm_gt)
            factor['mse_%d'%i] = 1
        return loss, factor
