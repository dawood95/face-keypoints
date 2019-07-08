import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models import resnet

class HRFPN34(nn.Module):

    def __init__(self, out_channels):

        super().__init__()

        base = resnet.resnet34(pretrained=True)
        base_channels = [64, 128, 256, 512]
        num_core = 128

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
            nn.Conv2d(base_channels[-1], num_core, 1),
            nn.BatchNorm2d(num_core),
            nn.ReLU(inplace = True),

            nn.Conv2d(num_core, num_core, 3, padding=1),
            nn.BatchNorm2d(num_core),
            nn.ReLU(inplace = True),
        )

        self.decoder  = nn.ModuleList([])
        self.lateral  = nn.ModuleList([])

        for in_channels in reversed(base_channels[:-1]):

            self.decoder.append(nn.Sequential(
                nn.Conv2d(num_core, num_core, 3, padding=1),
                nn.BatchNorm2d(num_core),
                nn.ReLU(inplace = True),
            ))

            self.lateral.append(nn.Sequential(
                nn.Conv2d(in_channels, num_core, 1),
                nn.BatchNorm2d(num_core),
                nn.ReLU(inplace = True),
            ))


        self.classifier = nn.Sequential(
            nn.Conv2d(num_core, num_core, 3, padding=1),
            nn.BatchNorm2d(num_core),
            nn.ReLU(inplace = True),
            nn.Conv2d(num_core, out_channels, 1),
        )

    def _add(self, x, y):
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

        laterals = []
        for i, l in enumerate(self.lateral):
            laterals.append(l(residuals[-(i + 2)]))

        x = self.decoder_start(x)

        for i, d in enumerate(self.decoder):
            x = self._add(x, laterals[i])
            x = d(x)
            if self.training:
                result.append(self.classifier(x))

        if not self.training:
            result.append(self.classifier(x))

        return result

    def calc_loss(self, preds, hm, mask):
        factor = {}
        loss = {}
        hm_mask = mask.repeat(1, hm.shape[1], 1, 1)
        mask_vals = list(range(1, int(mask.max().data.item()) + 1))
        for i, pred in enumerate(preds):
            _, _, h, w = pred.shape
            hm_gt    = F.interpolate(hm, (h, w), mode="bilinear", align_corners=True)
            mask_gt  = F.interpolate(hm_mask, (h, w))

            hm_pred  = pred[:, :-1]
            ae_pred  = pred[:, -1].unsqueeze(1)

            hm_pred[mask_gt == 0] = hm_gt[mask_gt == 0]

            diff = (hm_pred - hm_gt) ** 2

            mask = (hm_gt > 0).float()
            npos = int(mask.sum().data.item())
            pos_loss = (mask * diff).sum() / npos

            mask = 1 - mask
            nneg = min(npos * 3, diff.numel())
            neg_loss = (mask * diff).view(-1).topk(k = nneg, sorted=False)[0]
            neg_loss = neg_loss.sum() / nneg

            loss['mse_pos_%d'%i] = pos_loss
            loss['mse_neg_%d'%i] = neg_loss

            factor['mse_pos_%d'%i] = 1
            factor['mse_neg_%d'%i] = 1

            means = []
            loss['pull_loss_%d'%i] = 0
            loss['push_loss_%d'%i] = 0
            factor['pull_loss_%d'%i] = 1
            factor['push_loss_%d'%i] = 1
            for val in mask_vals:
                masked_pred = ae_pred[mask_gt[:, 0].unsqueeze(1) == val]
                mean = masked_pred.mean()
                means.append(mean)

                pull_loss = (masked_pred - mean) ** 2
                pull_loss = pull_loss.mean()

                loss['pull_loss_%d'%i] += pull_loss

            for m, mean1 in enumerate(means):
                for n, mean2 in enumerate(means):
                    if m == n: continue

                    diff = torch.clamp(1 - torch.abs(mean1 - mean2), min=0)
                    push_loss = diff
                    #x = (mean1 - mean2) ** 2
                    #x = -0.5 * x
                    #push_loss = torch.exp(x).mean()

                    loss['push_loss_%d'%i] += push_loss
            loss['push_loss_%d'%i] /= len(means)*2

        return loss, factor
