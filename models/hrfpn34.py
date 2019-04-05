import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models import resnet

class HRFPN34(nn.Module):

    def __init__(self, out_channels, smart_add=True, sigmoid=True):
        
        super().__init__()

        base = resnet.resnet34(pretrained=True)
        base_channels = [64, 128, 256, 512]

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder  = nn.ModuleList([
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        ])

        self.decoder_start = nn.Sequential(
            # lateral 1x1 connection
            nn.Conv2d(base_channels[-1], 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
        )

        self.decoder  = nn.ModuleList([])
        self.lateral  = nn.ModuleList([])
        self.upsample = nn.ModuleList([])

        for in_channels in reversed(base_channels[:-1]):

            self.decoder.append(nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),
            ))

            self.lateral.append(nn.Sequential(
                nn.Conv2d(in_channels, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),
            ))

            self.upsample.append(nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ))


        classifier = [
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, out_channels, 1)
        ]
        if sigmoid: classifier.append(nn.Sigmoid())
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.classifier = nn.Sequential(*classifier)
        self.smart_add = smart_add
        
    def _add(self, x, y):
        if not self.smart_add: return x + y
        
        if (x.numel() > y.numel()):
            x, y = y, x
        _, _, h, w = y.size()
        x = F.interpolate(x, size=(h, w))
        return x + y

    def forward(self, x):
        result = []

        x = self.in_block(x)
        residuals = []
        for e in self.encoder:
            x = e(x)
            residuals.append(x)
        x = self.decoder_start(x)

        down_path = zip(self.lateral, self.upsample, self.decoder)
        for i, (l, u, d) in enumerate(down_path):
            x = u(x)            
            x = self._add(x, l(residuals[-(i + 2)]))
            x = d(x)
            if self.training:
                result.append(self.classifier(x))

        if not self.training:
            result.append(self.classifier(x))
            
        return result

    def calc_loss(self, preds, hm):
        factor = {}
        loss = {}
        for i, pred in enumerate(preds):
            _, _, h, w = pred.shape
            hm_gt = F.interpolate(hm, (h, w), mode="bilinear", align_corners=True)
            loss['mse_%d'%i] = self.mse_loss(pred, hm_gt)
            factor['mse_%d'%i] = 1
        return loss, factor
