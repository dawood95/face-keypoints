from pathlib import Path
from torch.utils.data import DataLoader
from .ls3d import LS3D

def load_LS3D(train_root, val_root, image_size, batch_size, num_workers, cuda):

    train_dataset = LS3D(train_root, image_size, augment=True)
    val_dataset   = LS3D(val_root, image_size)

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


__all__ = ["load_LS3D"]
