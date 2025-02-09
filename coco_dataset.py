import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import pickle

from PIL import Image

from utils import iou
from config import COCO_original_classes_to_list_idx


class CachedCOCO:
    """ 
    Класс для кэширования объекта COCO.
    """
    def __init__(self, annotation_path):
        self.annotation_path = annotation_path
        self.cache_path = annotation_path + '.pickle'
        self.coco = self._load_coco()
        
    def _load_coco(self):
        if os.path.exists(self.cache_path):
            # Загружаем из кэша
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Создаем новый объект COCO
        coco = COCO(self.annotation_path)
        
        # Сохраняем в кэш
        with open(self.cache_path, 'wb') as f:
            pickle.dump(coco, f)
            
        return coco
    
    def __getattr__(self, name):
        # Делегируем все вызовы оригинальному объекту COCO
        return getattr(self.coco, name)

        
class COCOYOLODataset(Dataset):
    """
    A custom dataset class for YOLO object detection on the COCO dataset.
    
    This class handles loading images and their corresponding bounding box annotations,
    applying transformations, and preparing target tensors for YOLO training.

    Используется стандартная библиотека pytcocotools для загрузки данных из COCO dataset.
        данные берутся из instances_train2017.json, instances_val2017.json
        
        в instances_XXX2017.json есть значения:
        bbox x y width height. где x, y - минимальные координаты bbox
    
    Outputs:
        список для каждого масштаба (3шт или 3 HEAD модели (13х13,26х26, 52х52)).
        в каждом масштабе: тензор из anchor_num - количество anchorbox (3 например) 
        для каждой ячейки (13х13,26х26, 52х52) выдать 6 значений:
        [is_object, x, y, w, h, class]
        is_object - есть ли в ячейке объект
        x, y - координаты центра bbox в ячейке
        w, h - ширина и высота ячейки
        class - класс объекта (0...80)

        итого выход:
        [tensor[anchor_num, 13, 13, 6], tensor[anchor_num, 26, 26, 6], tensor[anchor_num, 52, 52, 6]]

    """

    def __init__(self, dir_path, anchors: list, scales: list, train=True, transform=None, num_images=None, input_coco=None):
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

        # Установим пути к директориям
        self.dir_path = dir_path

        if train:
            self.annotation_path = os.path.join(dir_path, "annotations\\instances_train2017.json")
            self.image_dir = os.path.join(dir_path, "train2017")
        else:
            self.annotation_path = os.path.join(dir_path, "annotations\\instances_val2017.json")
            self.image_dir = os.path.join(dir_path, "val2017")
        
        # Инициализируем coco API если не передался 
        self.coco = CachedCOCO(self.annotation_path)

        # Получаем список изображений и их id (list)
        self.img_ids= self.coco.getImgIds()

        self.transform = transform
        
        # Flatten and convert anchors to tensor
        self.anchors = torch.tensor(anchors)
        self.scales = scales
        
        # Set up anchor and scale properties
        self.num_anchor_in_scale = len(anchors[0])
        self.scale_numbers = len(self.scales)
        

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.img_ids)
    
    # Функция для преобразования аннотаций
    def _convert_original_class_to_list_idx(self, original_class_id):
        """
        Преобразует оригинальный id класса в индекс для модели
        """
        return COCO_original_classes_to_list_idx[original_class_id]

    def _load_image_and_boxes(self, idx: int ):
        # Получим id изображения
        img_id = self.img_ids[idx]
        # Получим изображение и его параметры в формате list(dict): {'license': 3,
                    #   'file_name': '000000184613.jpg',
                    #   'coco_url': 'http://images.cocodataset.org/train2017/000000184613.jpg',
                    #   'height': 336,
                    #   'width': 500,
                    #   'date_captured': '2013-11-14 12:36:29',
                    #   'flickr_url': 'http://farm3.staticflickr.com/2169/2118578392_1193aa04a0_z.jpg',
                    #   'id': 184613}]
        img_info = self.coco.loadImgs(img_id)
        img_width = img_info[0]['width']
        img_height = img_info[0]['height']
        image_path = os.path.join(self.image_dir, img_info[0]['file_name'])


        # Получим ids аннотаций к изображению
        anns_ids = self.coco.getAnnIds(imgIds=img_id)
        # Получим аннотации к изображению list[dict]
        # аннотации заданы в формате: {....,"category_id": xxx, 'bbox':[xmin, ymin, w, h] ,...} xmin, ymin, w,h - координаты bbox в ПИКСЕЛЯХ
        img_anns = self.coco.loadAnns(anns_ids)
        
        bboxes = []
        # Пройдемся по всем аннотациям изображения и преобразуем их в формат [x_center, y_center, w, h, class]. и занесем в список bboxes
        for i, ann in enumerate(img_anns):
            
            bbox = ann['bbox']
            
            # Переведем размеры bbox в относительные значения
            bbox = [bbox[0] / img_width, bbox[1] / img_height, bbox[2] / img_width, bbox[3] / img_height]

            # переведем формат bbox в формат [x_center, y_center, w, h, class]
            bbox = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]

            # Преобразовываем класс объекта номер класса по списку
            converted_class_id = self._convert_original_class_to_list_idx(ann['category_id'])
            # Добавим класс объекта
            bbox.append(converted_class_id)

            bboxes.append(bbox)
            if i>1:
                break

        image = np.array(Image.open(image_path).convert("RGB"))
        
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
        

        targets = self._create_target_tensor(bboxes)

        return image, targets
    


    

if __name__ == "__main__":
    from config import (
        
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
    PATH = "D:\\Learning\\Datasets\\coco2017"
    
    # Create the dataset
    def get_data_loaders(
                    path: str,
                    num_images=None,
                    train_transforms=TRAIN_TRANSFORMS,
                    test_transforms=TEST_TRANSFORMS,
                ):
        train_dataset = COCOYOLODataset(
            dir_path=path,
            anchors=ANCHORS,
            scales=SIZES,
            train=True,
            transform=train_transforms,
            num_images=num_images,
        )


    get_data_loaders(path=PATH, num_images=5)



