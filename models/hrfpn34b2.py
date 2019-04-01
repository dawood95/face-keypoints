import torch
from torch import nn
from torch.nn import functional as F
from .resnet import resnet34

class HRFPN34B2(nn.Module):
    NUM_KPT  = 68
    NUM_CORE = 256

    def __init__(self, base_pretrained=True, sigmoid=True, smart_add=True):
        super(HRFPN34B2, self).__init__()

        num_kpt  = HRFPN34B2.NUM_KPT + 1
        num_core = HRFPN34B2.NUM_CORE

        base = resnet34(pretrained=base_pretrained)
        base_channels = [64, 128, 256, 512]

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder = nn.ModuleList([
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4
        ])

        self.decoder_start = nn.Sequential(
            nn.Conv2d(base_channels[-1], num_core, 1),
            nn.BatchNorm2d(num_core),
            nn.ReLU(inplace=True)
        )

        self.lateral  = nn.ModuleList([])
        self.upsample = nn.ModuleList([])

        for in_channels in reversed(base_channels[:-1]):

            self.upsample.append(nn.Sequential(
                nn.Conv2d(num_core, num_core // 4, 1),
                nn.BatchNorm2d(num_core // 4),
                nn.ReLU(inplace = True),

                nn.ConvTranspose2d(num_core // 4, num_core // 4, 4,
                                   stride=2, padding=1, output_padding=0),
                nn.BatchNorm2d(num_core // 4),
                nn.ReLU(inplace = True),

                nn.Conv2d(num_core // 4, num_core, 1),
                nn.BatchNorm2d(num_core),
            ))

            self.lateral.append(nn.Sequential(
                nn.Conv2d(in_channels, num_core, 1),
                nn.BatchNorm2d(num_core),
            ))

        self.classifier = [
            nn.Conv2d(num_core, num_core // 4, 1),
            nn.BatchNorm2d(num_core // 4),
            nn.ReLU(inplace = True),

            nn.Conv2d(num_core // 4, num_core // 4, 3, padding=1),
            nn.BatchNorm2d(num_core // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_core // 4, num_core, 1),
            nn.BatchNorm2d(num_core),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_core, num_kpt, 1)
        ]
        if sigmoid: self.classifier.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*self.classifier)
        self.smart_add  = smart_add

    def _add(self, x, y):
        if not self.smart_add: return x + y
        if (x.numel() > y.numel()):
            x, y = y, x
        _, _, h, w = y.size()
        x = F.upsample(x, size=(h, w), mode='bilinear', align_corners=True)
        return x + y

    def forward(self, x):
        preds = []

        x = self.in_block(x)
        residuals = []
        for e in self.encoder:
            x = e(x)
            residuals.append(x)

        x = self.decoder_start(x)
        down_path = zip(self.lateral, self.upsample)
        for i, (l, u) in enumerate(down_path):
            x = u(x)
            r = l(residuals[-(i+2)])
            x = self._add(x, r)
            x = F.relu(x)
            c = self.classifier(x)
            preds.append(c)

        return preds

    @staticmethod
    def calc_loss(preds, hm_gt, mask):
        hm_loss = 0
        hm_mask = mask.unsqueeze(1).repeat(1, hm_gt.shape[1], 1, 1)
        for hm_pred in preds:
            b, c, h, w = hm_pred.shape
            _hm_gt   = F.upsample(hm_gt, (h, w), mode='bilinear', align_corners=True)
            _hm_mask = F.upsample(hm_gt, (h, w), mode='bilinear', align_corners=True)
            # hm_target = _hm_gt.clone()
            # hm_target[_hm_mask > 0] = hm_pred[_hm_mask > 0]
            hm_loss += ((_hm_gt - hm_pred) ** 2).sum() / hm_pred.shape[0]
        return hm_loss
