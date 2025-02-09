import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image

from utils import iou


"""
Используется набор PASCALVOC2012 в котором есть 2 директории и 3 csv файла:
|-VOC2012
|--images
|--labels
|--train.csv
|--test.csv
images - директория с изображениями jpg
labels - директория с файлами txt с информацией о bounding box формата:
    class x y width height, x, y - координаты центра bbox

    8 0.585 0.7306666666666667 0.122 0.3413333333333333
    8 0.41600000000000004 0.8453333333333333 0.176 0.288
    8 0.534 0.6546666666666666 0.108 0.27999999999999997

Задача:
на вход подается изображение и список bbox.

Выход:
список для каждого масштаба (3шт или 3 HEAD модели (13х13,26х26, 52х52)).
в каждом масштабе: тензор из anchor_num - количество anchorbox (3 например) 
для каждой ячейки (13х13,26х26, 52х52) выдать 6 значений:
[is_object, x, y, w, h, class]
is_object - есть ли в ячейке объект
x, y - координаты ячейки
w, h - ширина и высота ячейки
class - класс объекта (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)

итого выход:
[tensor[anchor_num, 13, 13, 6], tensor[anchor_num, 26, 26, 6], tensor[anchor_num, 52, 52, 6]]
""" 


        
class VOCYOLODataset(Dataset):
    """
    A custom dataset class for YOLO object detection on the PASCAL VOC dataset.
    
    This class handles loading images and their corresponding bounding box annotations,
    applying transformations, and preparing target tensors for YOLO training.

    Используется набор PASCALVOC2012 в котором есть 2 директории и 3 csv файла:
        |-VOC2012
        |--images
        |--labels
        |--train.csv
        |--test.csv
        images - директория с изображениями jpg
        labels - директория с файлами txt с информацией о bounding box формата:
            class x y width height, x, y - координаты центра bbox
    
    Outputs:
        список для каждого масштаба (3шт или 3 HEAD модели (13х13,26х26, 52х52)).
        в каждом масштабе: тензор из anchor_num - количество anchorbox (3 например) 
        для каждой ячейки (13х13,26х26, 52х52) выдать 6 значений:
        [is_object, x, y, w, h, class]
        is_object - есть ли в ячейке объект
        x, y - координаты ячейки
        w, h - ширина и высота ячейки
        class - класс объекта (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)

        итого выход:
        [tensor[anchor_num, 13, 13, 6], tensor[anchor_num, 26, 26, 6], tensor[anchor_num, 52, 52, 6]]

    """

    def __init__(self, dir_path, anchors: list, scales: list, train=True, transform=None, num_images=None):
        """
        Initialize the dataset.
        
        :param dir_path: Path to the dataset directory
        :param anchors: List of anchor box dimensions for each scale
        :param scales: List of output scales (e.g., [13, 26, 52])
        :param train: Whether to use training or testing data
        :param transform: Optional transformations to apply to images and bounding boxes
        :param num_images: Number of images to load (None for all)
        """
         
        super().__init__()
        
         # Validate that anchors and scales have the same length
        try:
            len(anchors) == len(scales)
        except:
            print("Anchors and scales must have the same length")
        
        # Set up file paths
        self.dir_path = dir_path
        self.csv_file_path = os.path.join(dir_path, "train.csv" if train else "test.csv")
        self.annotations_dir = os.path.join(dir_path, "labels")
        self.image_dir = os.path.join(dir_path, "images")

        # Load annotations
        self.annotations = pd.read_csv(self.csv_file_path, nrows=num_images)
        self.transform = transform
        
        # Flatten and convert anchors to tensor
        self.anchors = torch.tensor(anchors)
        self.scales = scales
        
        # Set up anchor and scale properties
        self.num_anchor_in_scale = len(anchors[0])
        self.scale_numbers = len(self.scales)

    def _load_image_and_boxes(self, idx: int):
        """Load image and bounding boxes from file for given index."""
        image_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        annotation_path = os.path.join(self.annotations_dir, self.annotations.iloc[idx, 1])

        image = np.array(Image.open(image_path).convert("RGB"))
        # Format: [class, x_center, y_center, width, height] converts to [x_center, y_center, width, height, class]
        bboxes = np.roll(np.loadtxt(annotation_path, ndmin=2), -1, axis=1).tolist()

        return image, bboxes
        
    def _create_target_tensor(self, bboxes: list) -> list[torch.Tensor]:
        """Create target tensor for all scales from bounding boxes"""
        # Initialize target tensors for each scale
        targets = [torch.zeros(size=(self.num_anchor_in_scale, scale, scale, 6))  for scale in self.scales]
        
        # Process each bounding box
        for bbox in bboxes: 
            
            # Find the best anchor for each scale (i.e. with the best iou)
            iou_anchors = iou(torch.tensor(bbox[2:4]),  self.anchors,  is_pred=False) 
            anchor_pos_in_scale = torch.argmax(iou_anchors, dim=1) 

            self._assign_box_to_targets(bbox=bbox, anchor_pos_in_scale=anchor_pos_in_scale, targets=targets)

        return targets
    
    def _assign_box_to_targets(self, bbox: list, anchor_pos_in_scale: torch.Tensor, targets: list[torch.Tensor]) -> None:
        """Assign a bounding box to target tensors for all scales"""
        x_center, y_center, bbox_w, bbox_h, obj_class_id  = bbox            
            
        # Process the bounding box for each scale
        for scale_number, cell_size in enumerate(self.scales):
            # Расчитывается координата центра bbox относительно ячейки
            x_cell_pos, y_cell_pos = int(x_center * cell_size), int(y_center * cell_size)

            anchor_num_for_box_in_scale = anchor_pos_in_scale[scale_number]
            
            # Calculate bounding box coordinates relative to the cell
            x_center_in_cell  = cell_size * x_center - x_cell_pos
            y_center_in_cell = cell_size * y_center - y_cell_pos
            width_for_cell  = bbox_w * cell_size
            height_for_cell = bbox_h * cell_size

            coordinates = torch.tensor([x_center_in_cell, y_center_in_cell, width_for_cell, height_for_cell])
            
            # Assign the bounding box to the target if the cell is empty.
            # Only assign if cell is empty
            if targets[scale_number][anchor_num_for_box_in_scale, x_cell_pos, y_cell_pos, 0] == 0:
                targets[scale_number][anchor_num_for_box_in_scale, x_cell_pos, y_cell_pos, 0] = 1
                # в позицию 1-4 targets pаписывается 
                targets[scale_number][anchor_num_for_box_in_scale, x_cell_pos, y_cell_pos, 1:5] = coordinates
                # в позицию 5 targets pаписывается класс bbox
                targets[scale_number][anchor_num_for_box_in_scale, x_cell_pos, y_cell_pos, 5] = torch.tensor(obj_class_id)


    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        :param idx: Index of the item to retrieve
        :return: Tuple of (transformed_image, targets)
            targets: [tensor[anchor_num, 13, 13, 6], tensor[anchor_num, 26, 26, 6], tensor[anchor_num, 52, 52, 6]]
        """
        # Load image and bounding boxes
        
        image, bboxes = self._load_image_and_boxes(idx)

        # Apply transformations if any
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        # print(bboxes)

        targets = self._create_target_tensor(bboxes)

        return image, targets
    

if __name__ == "__main__":
    pass
