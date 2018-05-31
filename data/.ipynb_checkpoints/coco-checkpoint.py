import torch
import random
import numpy as np

from PIL import Image
from pathlib import Path
from torch.utils import data
from torchvision.transforms import functional as T
from pycocotools.coco import COCO as pyCOCO

class COCO(data.Dataset):
    AUG_ANGLE = 60
    FLIP_SRC  = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    FLIP_TGT  = [2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]

    def __init__(self, root, annFile, image_size=512, augment=True, sigma=7.):
        # inititalize pyCOCO
        self.coco     = pyCOCO(Path(annFile).as_posix())
        self.catIds   = self.coco.getCatIds(catNms = ['person'])
        self.imgIds   = self.coco.getImgIds(catIds = self.catIds)
        self.imgs     = self.coco.loadImgs(self.imgIds)
        self.skeleton = self.coco.loadCats(self.catIds)[0]['skeleton']
        self.img_pth  = lambda x : (Path(root) / x).as_posix()

        # NOTE: Make sure VGG-19 needs normalization
        norm_mean = [0.485, 0.456, 0.406]
        norm_std  = [0.229, 0.224, 0.225]
        self.normalize = lambda x : T.normalize(x, norm_mean, norm_std)

        self.sigma      = sigma
        self.augment    = augment
        self.image_size = image_size

    def __getitem__(self, idx):
        # Load img and anns
        img    = self.imgs[idx]
        img_id = img['id']
        annIds = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds)

        img  = Image.open(self.img_pth(img['file_name'])).convert('RGB')
        anns = self.coco.loadAnns(annIds)

        # Create loss mask
        ow, oh    = img.size
        loss_mask = np.zeros((oh, ow), dtype=np.float)
        for ann in anns:
            if ann['num_keypoints'] == 0: continue
            loss_mask = np.maximum(loss_mask, self.coco.annToMask(ann))
        loss_mask = Image.fromarray(loss_mask)

        # Scale img and loss mask
        img       = T.resize(img, self.image_size)
        loss_mask = T.resize(loss_mask, self.image_size)
        nw, nh    = img.size
        sx, sy    = nw / ow, nh / oh

        # Create augment params
        angle = random.randrange(-COCO.AUG_ANGLE, COCO.AUG_ANGLE) if self.augment else 0
        flip  = random.random() > 0.5 and self.augment

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
        h, w   = h, w
        hm   = np.zeros((17 + 1, h, w), dtype=np.float)
        jm_x = np.zeros((19, h, w), dtype=np.float)
        jm_y = np.zeros((19, h, w), dtype=np.float)
        cnts = np.zeros(    (h, w), dtype=np.float)

        base = np.indices((h, w), dtype=np.float)
        kpt_base = np.tile(base, (17, 1, 1))

        for ann in anns:
            if ann['num_keypoints'] == 0: continue

            keypoints = np.array(ann['keypoints']).reshape(-1, 3)

            x, y, vis = keypoints[:, 0].copy(), keypoints[:, 1].copy(), keypoints[:, 2].copy()
            keypoints[:, 0] = y * sy
            keypoints[:, 1] = x * sx
            keypoints[:, 2] = 1

            if flip:
                keypoints[:, 1]             = nw - keypoints[:, 1]
                keypoints[COCO.FLIP_SRC, :] = keypoints[COCO.FLIP_TGT, :]
                vis[COCO.FLIP_SRC]          = vis[COCO.FLIP_TGT]

            # Affine transforms
            keypoints = (affine_M @ keypoints.T).T
            keypoints[:, [0, 1]] = keypoints[:, [1, 0]]

            # Random crop translation and reconstruct original struct
            keypoints[:, 0] -= x1
            keypoints[:, 1] -= y1
            keypoints = np.concatenate([keypoints, vis[:, np.newaxis]], axis=1)

            # kpt_arrays for hm
            kpt_arrays = []
            for x, y, v in keypoints:
                kpt_arr = np.zeros_like(base)
                if v != 2:
                    kpt_arr[:] = np.inf
                else:
                    kpt_arr[0, :] = y
                    kpt_arr[1, :] = x
                kpt_arrays.append(kpt_arr)

            kpt_arrays = np.concatenate(kpt_arrays, axis=0)
            kpt_arrays = (kpt_arrays - kpt_base) ** 2

            for i in range(17):
                dst = np.exp(-np.sum(kpt_arrays[2*i:2*(i + 1)], axis=0) / (2 * (self.sigma ** 2)))
                hm[i] = np.maximum(hm[i], dst)

            # jointmap
            for i, (src, dst) in enumerate(self.skeleton):
                src = keypoints[src - 1]
                dst = keypoints[dst - 1]
                if src[-1] != 2 or dst[-1] != 2: continue

                pt1_x, pt1_y = src[:2]
                pt2_x, pt2_y = dst[:2]
                vec_x, vec_y = pt2_x - pt1_x, pt2_y - pt1_y

                norm = np.sqrt((vec_x ** 2) + (vec_y ** 2))
                if norm == 0: continue
                vec_x, vec_y = vec_x / norm, vec_y / norm

                vec_arr = np.zeros_like(base)
                vec_arr[0, :] = pt1_y
                vec_arr[1, :] = pt1_x
                diff = base - vec_arr
                dot_prod = (diff[0] * vec_y) + (diff[1] * vec_x)

                ort_prod = dot_prod / (vec_x ** 2 + vec_y ** 2)
                ort_prod = ((diff[0] - (ort_prod * vec_y)) * diff[0]) + ((diff[1] - (ort_prod * vec_x)) * diff[1])

                ort_prod = abs(ort_prod) < (self.sigma ** 2)
                dot_prod = (dot_prod <= norm) * (dot_prod > 0)
                mask     = ort_prod * dot_prod

                jm_x[i][mask] += vec_x
                jm_y[i][mask] += vec_y
                cnts[mask] += 1

        hm[-1, :, :] = 1. - np.amax(hm[:-1, :, :], axis=0)
        jm_x[:, cnts > 0] /= cnts[cnts > 0]
        jm_y[:, cnts > 0] /= cnts[cnts > 0]

        gt_hm = torch.Tensor(hm)
        gt_jm = torch.cat((torch.Tensor(jm_x), torch.Tensor(jm_y)), dim=0)

        return img, gt_hm, gt_jm, loss_mask, img_id

    def __len__(self):
        return len(self.imgs)
