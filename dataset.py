import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image

from iou_utils import iou


"""
Используется набор PASCALVOC2012 в котором есть 2 директории и 3 csv файла:
|-VOC2012
|--images
|--labels
|--train.csv
|--test.csv
images - директория с изображениями jpg
labels - директория с файлами txt с информацией о bounding box формата:
    class x y width height
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
        self.anchor_numbers_in_scale = len(anchors[0])
        self.scale_numbers = len(self.scales)
        

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
        image_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        annotation_path = os.path.join(self.annotations_dir, self.annotations.iloc[idx, 1])
        
        image = np.array(Image.open(image_path).convert("RGB"))
        bboxes = np.roll(np.loadtxt(annotation_path, ndmin=2), -1, axis=1).tolist()
        
        # Apply transformations if any
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Initialize target tensors for each scale
        targets = [torch.zeros(size=(self.anchor_numbers_in_scale, scale, scale, 6))  for scale in self.scales]
        
        # Process each bounding box
        for bbox in bboxes: 
            # Calculate iou of bounding box with anchor boxes 
            iou_anchors = iou(torch.tensor(bbox[2:4]),  self.anchors,  is_pred=False) 

            bbox_x, bbox_y, bbox_w, bbox_h, obj_class  = bbox            


            # Find the best anchor for each scale (i.e. with the best iou)
            anchor_pos_in_scale = torch.argmax(iou_anchors, dim=1) #size = (scale_numbers)
            
            # Process the bounding box for each scale
            for scale_number, cell_size in enumerate(self.scales):
                # Calculate the cell position of the bounding box center
                x_cell_position, y_cell_position = int(bbox_x * cell_size), int(bbox_y * cell_size)

                anchor_num_for_box_in_scale = anchor_pos_in_scale[scale_number]
                
                # Calculate bounding box coordinates relative to the cell
                x_center_in_cell, y_center_in_cell = cell_size * bbox_x - x_cell_position, cell_size * bbox_y - y_cell_position
                width_for_cell, height_for_cell = (bbox_w * cell_size, bbox_h * cell_size)
                coordinates = torch.tensor([x_center_in_cell, y_center_in_cell, width_for_cell, height_for_cell])
                
                # Assign the bounding box to the target if the cell is empty.
                if targets[scale_number][anchor_num_for_box_in_scale, x_cell_position, y_cell_position, 0] == 0:
                    targets[scale_number][anchor_num_for_box_in_scale, x_cell_position, y_cell_position, 0] = 1
                    # в позицию 1-4 targets pаписывается 
                    targets[scale_number][anchor_num_for_box_in_scale, x_cell_position, y_cell_position, 1:5] = coordinates
                    # в позицию 5 targets pаписывается класс bbox
                    targets[scale_number][anchor_num_for_box_in_scale, x_cell_position, y_cell_position, 5] = torch.tensor(obj_class)

        return image, targets
    

if __name__ == "__main__":
    from config import PATH, ANCHORS, SIZES
    train_dataset = VOCYOLODataset(dir_path=PATH, anchors=ANCHORS, scales=SIZES, train=True, transform=None, num_images=10)