
import torch
import random
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils import data, serialization
from torchvision.transforms import functional as T
from pycocotools.coco import COCO as pyCOCO

class LS3D(data.Dataset):
    AUG_ANGLE = 60

    def __init__(self, root, image_size=256, image_type='jpg', augment=True, sigma=7.):
        # initialize LS3D imgs and anns
        root = Path(root)
        self.imgs = []
        self.anns = []
        for img in root.glob('**/*.%s'%image_type):
            ann = img.as_posix()[:-3]+'t7'
            if Path(ann).exists():
                self.imgs.append(img.as_posix())
                self.anns.append(ann)

        norm_mean = [0.485, 0.456, 0.406]
        norm_std  = [0.229, 0.224, 0.225]
        self.normalize = lambda x : T.normalize(x, norm_mean, norm_std)

        self.sigma      = sigma
        self.augment    = augment
        self.image_size = image_size

    def __getitem__(self, idx):
        # Load img and anns
        img = self.imgs[idx]
        ann = self.anns[idx]

        img = Image.open(img).convert('RGB')
        ann = serialization.load_lua(ann)

        # Create loss mask
        ow, oh    = img.size
        loss_mask = np.zeros((oh, ow), dtype=np.float)
        x1, y1, x2, y2 = min(ann[:, 0]), min(ann[:, 1]), max(ann[:, 0]), max(ann[:, 1])
        h, w = y2 - y1, x2 - x1
        x1  = int(max(0, x1.item() - w * 0.5))
        y1  = int(max(0, y1.item() - h * 0.5))
        x2  = int(min(ow, x2.item() + w * 0.5))
        y2  = int(min(oh, y2.item()) + h * 0.5)
        loss_mask[y1:y2, x1:x2] = 1
        loss_mask = Image.fromarray(loss_mask)

        # Scale img and loss mask
        img       = T.resize(img, self.image_size)
        loss_mask = T.resize(loss_mask, self.image_size)
        nw, nh    = img.size
        sx, sy    = nw / ow, nh / oh

        # Create augment params
        angle = random.randrange(-LS3D.AUG_ANGLE, LS3D.AUG_ANGLE) if self.augment else 0
        flip  = False # TODO

        # Perform augmentations to images and loss masks
        img       = T.hflip(img) if flip else img
        loss_mask = T.hflip(loss_mask) if flip else loss_mask
        img       = T.affine(img, angle, [0, 0], 1, 0)
        loss_mask = T.affine(loss_mask, angle, [0, 0], 1, 0)

        # Compute Affine matrix for keypoints
        affine_M = T._get_inverse_affine_matrix((nh / 2, nw / 2), angle, [0, 0], 1, 0)
        affine_M = np.array(affine_M).reshape(2, 3)

        # Random crop co-ordinates
        if self.augment:
            x1 = random.randrange(0, nw - self.image_size) if nw > nh else 0
            x2 = x1 + self.image_size
            y1 = random.randrange(0, nh - self.image_size) if nh > nw else 0
            y2 = y1 + self.image_size
        else:
            x1, y1 = 0, 0
            x2, y2 = nw, nh

        # PIL -> Numpy
        img       = T.to_tensor(img)
        loss_mask = T.to_tensor(loss_mask)

        # Random crop image and loss mask
        img       = img[:, y1:y2, x1:x2]
        loss_mask = loss_mask[:, y1:y2, x1:x2]

        # hack for now
        loss_mask = T.to_pil_image(loss_mask)
        loss_mask = T.resize(loss_mask, self.image_size)
        loss_mask = T.to_tensor(loss_mask)[0]

        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        # Generate heatmap
        _, h, w = img.shape
        h, w   = h // 2, w // 2
        hm   = np.zeros((68 + 1, h, w), dtype=np.float)

        base = np.indices((h, w), dtype=np.float)
        kpt_base = np.tile(base, (68, 1, 1))

        keypoints = np.ones((68, 3))
        keypoints[:, 0] = ann[:, 1] * sy
        keypoints[:, 1] = ann[:, 0] * sx

        if flip:
            keypoints[:, 1]             = nw - keypoints[:, 1]
            keypoints[LS3D.FLIP_SRC, :] = keypoints[LS3D.FLIP_TGT, :]
            vis[LS3D.FLIP_SRC]          = vis[LS3D.FLIP_TGT]

        # Affine transforms
        keypoints = (affine_M @ keypoints.T).T
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]

        # Random crop translation and reconstruct original struct
        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1

        # kpt_arrays for hm
        kpt_arrays = []
        for x, y in keypoints:
            kpt_arr = np.zeros_like(base)
            kpt_arr[0, :] = y // 2
            kpt_arr[1, :] = x // 2
            kpt_arrays.append(kpt_arr)

        kpt_arrays = np.concatenate(kpt_arrays, axis=0)
        kpt_arrays = (kpt_arrays - kpt_base) ** 2

        for i in range(68):
            dst = np.exp(-np.sum(kpt_arrays[2*i:2*(i + 1)], axis=0) / (2 * (self.sigma ** 2)))
            hm[i] = np.maximum(hm[i], dst)

        hm[-1, :, :] = 1. - np.amax(hm[:-1, :, :], axis=0)
        gt_hm = torch.Tensor(hm)

        return self.normalize(img), gt_hm, loss_mask

    def __len__(self):
        return len(self.imgs)
