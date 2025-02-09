from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import VOCYOLODataset
from coco_dataset import COCOYOLODataset

from config import (
    PATH,
    ANCHORS,
    IMAGE_SIZE,
    SIZES,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_TRANSFORMS,
    TEST_TRANSFORMS,
    NEED_TO_CHANGE_LR,
)

def get_data_loaders(
    path: str,
    num_images=None,
    train_transforms=TRAIN_TRANSFORMS,
    test_transforms=TEST_TRANSFORMS,
    dataset_type="voc"

):
    if dataset_type == "coco":
        train_dataset = COCOYOLODataset(
            dir_path=path,
            anchors=ANCHORS,
            scales=SIZES,
            train=True,
            transform=train_transforms,
            num_images=num_images,
        )
        test_dataset = COCOYOLODataset(
            dir_path=path,
            anchors=ANCHORS,
            scales=SIZES,
            train=False,
            transform=test_transforms,
            num_images=num_images,
        )

    else:
        train_dataset = VOCYOLODataset(
            dir_path=path,
            anchors=ANCHORS,
            scales=SIZES,
            train=True,
            transform=train_transforms,
            num_images=num_images,
        )

        test_dataset = VOCYOLODataset(
            dir_path=path,
            anchors=ANCHORS,
            scales=SIZES,
            train=False,
            transform=test_transforms,
            num_images=num_images,
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    return train_loader, test_loader



