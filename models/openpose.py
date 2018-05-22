import torch

from torch import nn
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
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        ))

        # List of all stages
        self.stages = nn.ModuleList([])

        # Put in first stage since it is different
        self.stages.append(nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, 1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(512, num_kpt, 1)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, 1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(512, num_paf, 1)
            )
        ]))

        inc = 128 + num_kpt + num_paf
        for _ in range(5):
            self.stages.append(nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(inc, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 1, padding=0), nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_kpt, 1)
                ),
                nn.Sequential(
                    nn.Conv2d(inc, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 7, padding=3), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 1, padding=0), nn.ReLU(inplace=True),
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
            hm_pred = hm(x)
            jm_pred = jm(x)
            preds.append((hm_pred, jm_pred))
            x = torch.cat((hm_pred, jm_pred, features), dim=1)

        return preds

    @staticmethod
    def calc_loss(preds, hm_gt, jm_gt, mask):
        loss = 0
        for pred in preds:
            hm_pred, jm_pred = pred
            hm_loss = ((hm_pred.cpu() - hm_gt) ** 2)[:, :, mask > 0].sum()
            jm_loss = ((jm_pred.cpu() - jm_gt) ** 2)[:, :, mask > 0].sum()
            loss += hm_loss + jm_loss
        return loss
