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

    def __init__(self, root, image_size=256, num_inst=3, sigma=7.):

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
            iaa.Resize(image_size),
        ])

        self.data  = data
        self.sigma = sigma
        self.aug   = aug

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
            aug = self.aug.to_deterministic()
            img = aug.augment_image(img)
            ann = aug.augment_keypoints([ann])[0].keypoints

            # Create heatmap
            h, w, _ = img.shape
            hm = np.zeros((len(ann), h, w), dtype=np.float)
            base = np.indices((h, w), dtype=np.float)
            kpt_base = np.tile(base, (len(ann), 1, 1))

            kpt_arrays = []
            for i in range(len(ann)):
                kpt_arr = np.zeros_like(base)
                # TODO: Check if this is flipped
                kpt_arr[0, :] = kpt[i].y
                kpt_arr[1, :] = kpt[i].x
                kpt_arrays.append(kpt_arr)
            kpt_arrays = np.concatenate(kpt_arrays, axis=0)
            kpt_arrays = (kpt_arrays - kpt_base) ** 2

            for i in range(len(ann)):
                sig = 2 * (self.sigma ** 2)
                dst = np.exp(-np.sum(kpt_arrays[2*i:2*(i+1)], axis=0) / sig)
                hm[i] = np.maximum(hm[i], dst)

            imgs.append(img)
            hmps.append(hm)

        imgs = np.concatenate(imgs, 2)
        hmps = np.concatenate(hmps, 2)

        imgs = torch.from_numpy(imgs)
        hmps = torch.from_numpy(hmps)

        imgs = imgs.permute(2, 0, 1).contiguous()
        imgs = self.normalize(imgs)

        return imgs, hmps

    def __len__(self):
        return len(self.data)
