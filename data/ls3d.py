import torch
import torchfile
import numpy as np
import imgaug as ia

from PIL                    import Image
from imgaug                 import augmenters as iaa
from random                 import shuffle
from pathlib                import Path
from itertools              import chain
from torch.utils.data       import Dataset
from torchvision.transforms import functional as T

class LS3D(Dataset):

    def __init__(self, root, image_size=256, num_inst=3, sigma=7., augment=False):
        root = Path(root)

        data = []
        imgs = chain(root.glob("**/*.jpg"), root.glob("**/*.png"))
        for img_path in imgs:
            img_path = img_path.as_posix()
            ann_path = img_path.replace('.jpg', '.t7').replace('.png', '.t7')
            if Path(ann_path).exists():
                data.append((img_path, ann_path))
        shuffle(data)
        data = [data[i:i+num_inst] for i in range(len(data)//num_inst)]

        norm_mean = [0.485, 0.456, 0.406]
        norm_std  = [0.229, 0.224, 0.225]
        self.normalize = lambda x : T.normalize(x, norm_mean, norm_std)

        aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale=(0.5, 1.5), rotate=(-60, 60)),
        ])
        resize = iaa.Resize(image_size)
        
        self.data    = data
        self.aug     = aug
        self.resize  = resize
        self.sigma   = sigma
        self.augment = augment
        
        size = 4 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        self.g = torch.Tensor(np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2))))

    def __getitem__(self, idx):
        imgs = []
        hmps = []

        data = self.data[idx]
        for img, ann in data:
            # Load image and keypoints
            img = np.array(Image.open(img).convert('RGB'))
            ann = torchfile.load(ann).astype(np.int32)
            ann = [ia.Keypoint(x, y) for (x, y) in ann]
            ann = ia.KeypointsOnImage(ann, img.shape)

            # Augment image and keypoints
            if self.augment:
                aug = self.aug.to_deterministic()
                img = aug.augment_image(img)
                ann = aug.augment_keypoints([ann])[0]
                
            img = self.resize.augment_image(img)
            ann = self.resize.augment_keypoints([ann])[0]
            ann = ann.keypoints

            # Create heatmap
            h, w, _ = img.shape
            hm = np.zeros((len(ann), h, w), dtype=np.float32)
            sigma = self.sigma
            for i in range(len(ann)):
                x, y = ann[i].x, ann[i].y

                ul = [int(x - 2 * sigma), int(y - 2 * sigma)]
                br = [int(x + 2 * sigma + 1), int(y + 2 * sigma + 1)]
                if (ul[0] >= w or ul[1] >= h or
                    br[0] < 0 or br[1] < 0):
                    continue

                g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
                img_x = max(0, ul[0]), min(br[0], w)
                img_y = max(0, ul[1]), min(br[1], h)
                if (img_x[1] - img_x[0] <= 0 or img_y[1] - img_y[0] <= 0 or
                    g_x[1] - g_x[0] <= 0 or g_y[1] - g_y[0] <= 0):
                    continue
                hm[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = self.g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
            imgs.append(img)
            hmps.append(hm)

        imgs = np.concatenate(imgs, 1)
        imgs = imgs.astype(np.float32) / 255.
        hmps = np.concatenate(hmps, 2)

        imgs = torch.from_numpy(imgs)
        hmps = torch.from_numpy(hmps)

        imgs = imgs.permute(2, 0, 1).contiguous()
        imgs = self.normalize(imgs)

        return imgs, hmps

    def __len__(self):
        return len(self.data)
