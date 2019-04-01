from pathlib import Path
from torch.utils.data import DataLoader
from .ls3d import LS3D

def get_LS3D(train_root, val_root, train_image_type, val_image_type,
             batch_size, num_workers, cuda):

    train_dataset = LS3D(train_root, image_type=train_image_type)
    val_dataset   = LS3D(val_root  , image_type=val_image_type, augment=False)

    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = cuda
    )

    val_loader = DataLoader(
        dataset     = val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = cuda
    )

    return train_loader, val_loader


__all__ = ["get_COCO"]
