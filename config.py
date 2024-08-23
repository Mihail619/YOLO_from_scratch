import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import os

os.environ['ALBUMENTATIONS_DISABLE_CHECKING_VERSION'] = '1'

NUM_EPOCHS = 1

PATH = "D:\\Learning\\Datasets\\PASCAL_VOC"
ANCHORS = [
    ((0.28, 0.22), (0.38, 0.48), (0.9, 0.78)),
    ((0.07, 0.15), (0.15, 0.11), (0.14, 0.29)),
    ((0.02, 0.03), (0.04, 0.07), (0.08, 0.06)),
]  # Note these have been rescaled to be between [0, 1]
IMAGE_SIZE = 416
SIZES = [52, 26, 13]
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
NUM_CLASSES = 20
BACKBONE_NUM_CHANNELS = [128, 256, 512]
NUMBER_BLOCKS_LIST = [3, 4, 4, 3]


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



scale = 1.2

train_transforms = A.Compose(
    [
        # A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
            p=1.0
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Affine(shear=15, p=0.5, mode=0),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        # A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, 
            min_width=IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
            p=1.0

        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)