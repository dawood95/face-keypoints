from pathlib import Path
from torch.utils.data import DataLoader
from .coco import COCO

from pycocotools.coco import COCO as eCOCO

def get_COCO(data_root, batch_size, num_workers, cuda):
    '''
    Not using for now. Using PCOCO instead
    '''
    train_annFile = Path(data_root) / 'annotations' / 'person_keypoints_train2017.json'
    val_annFile = Path(data_root) / 'annotations' / 'person_keypoints_val2017.json'

    train_dataset = COCO(Path(data_root) / 'train2017',
                         train_annFile,
    )
    val_dataset = COCO(Path(data_root) / 'val2017',
                       val_annFile,
                       augment=False
    )

    num_keypoints, num_connections = train_dataset.target_channels()

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
        num_workers = 1,#num_workers,
        pin_memory  = cuda
    )

    cocoGt = eCOCO(val_annFile.as_posix())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)
    
    return train_loader, val_loader, (cocoGt, imgIds), (num_keypoints, num_connections)


__all__ = ["get_COCO"]
