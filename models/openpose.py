import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg

class Openpose(nn.Module):
    NUM_KPT = 17
    NUM_PAF = 19

    def __init__(self, base_pretrained=True):
        super(Openpose, self).__init__()

        num_kpt = Openpose.NUM_KPT + 1
        num_paf = Openpose.NUM_PAF * 2

        vgg19 = vgg.vgg19(pretrained=base_pretrained)

        # First 10 layers of VGG-19
        self.vgg_base = vgg19.features[:27]
        self.vgg_base.add_module('extra_layers', nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ))

        # List of all stages
        self.stages = nn.ModuleList([])

        # Put in first stage since it is different
        self.stages.append(nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, 1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, num_kpt, 1)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, 1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, num_paf, 1)
            )
        ]))

        inc = 128 + num_kpt + num_paf
        for _ in range(5):
            self.stages.append(nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(inc, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 1, padding=0),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_kpt, 1)
                ),
                nn.Sequential(
                    nn.Conv2d(inc, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 1, padding=0),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_paf, 1)
                )
            ]))

    def forward(self, x):
        # output preds
        preds = []

        # vgg base
        features = self.vgg_base(x)

        # openpose stages
        x = features
        for hm, jm in self.stages:
            hm_pred = F.sigmoid(hm(x))
            jm_pred = F.tanh(jm(x))
            preds.append((hm_pred, jm_pred))
            x = torch.cat((hm_pred, jm_pred, features), dim=1)

        return preds

    @staticmethod
    def calc_loss(preds, hm_gt, jm_gt, mask):
        hm_loss = 0
        jm_loss = 0

        hm_mask = mask.unsqueeze(1).repeat(1, hm_gt.shape[1], 1, 1)
        jm_mask = mask.unsqueeze(1).repeat(1, jm_gt.shape[1], 1, 1)
        for pred in preds:
            hm_pred, jm_pred = pred

            b, c, h, w = hm_pred.shape

            if hm_pred.shape != hm_gt.shape:
                hm_gt = F.upsample(hm_gt, (h, w), mode='bilinear', align_corners=True)
                jm_gt = F.upsample(jm_gt, (h, w), mode='bilinear', align_corners=True)
                hm_mask = F.upsample(hm_mask, (h, w), mode='bilinear', align_corners=True)
                jm_mask = F.upsample(jm_mask, (h, w), mode='bilinear', align_corners=True)

            hm_target = hm_gt.clone()
            jm_target = jm_gt.clone()
            hm_target[hm_mask > 0] = hm_pred[hm_mask > 0]
            jm_target[jm_mask > 0] = jm_pred[jm_mask > 0]

            hm_loss += ((hm_pred - hm_gt) ** 2).sum() / hm_pred.shape[0]
            jm_loss += ((jm_pred - jm_gt) ** 2).sum() / jm_pred.shape[0]

        return hm_loss, jm_loss
