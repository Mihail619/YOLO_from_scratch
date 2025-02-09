import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from torch import nn
from torch.optim import Adam
import torchvision.ops as ops


from collections import Counter
from logger_config import logger


from config import (
    DATASET_TYPE,
    NUM_CLASSES,
    IMAGE_SIZE,
    SIZES,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_TRANSFORMS,
    LOAD_MODEL,
    NEED_TO_CHANGE_LR,
    LEARNING_RATE
)
from my_yolo_model import YOLO


# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # IoU score based on width and height of bounding boxes

        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(
            box1[..., 1], box2[..., 1]
        )

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score


# Non-maximum suppression function to remove overlapping bounding boxes
def custom_nms(bboxes, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    # bboxes = [box for box in bboxes if box[1] > threshold]
    mask = bboxes[..., 1] > threshold
    bboxes = bboxes[mask]
    bboxes = bboxes.tolist()

    # Sort the bounding boxes by confidence in descending order.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # Initialize the list of bounding boxes after non-maximum suppression.
    bboxes_nms = []

    while bboxes:
        # Get the first bounding box.
        first_box = bboxes.pop(0)

        # Iterate over the remaining bounding boxes.
        for box in bboxes:
            # If the bounding boxes do not overlap or if the first bounding box has
            # a higher confidence, then add the second bounding box to the list of
            # bounding boxes after non-maximum suppression.
            if (
                box[0] != first_box[0]
                or iou(
                    torch.tensor(first_box[2:]),
                    torch.tensor(box[2:]),
                )
                < iou_threshold
            ):
                # Check if box is not in bboxes_nms
                if box not in bboxes_nms:
                    # Add box to bboxes_nms
                    bboxes_nms.append(box)

    # Return bounding boxes after non-maximum suppression.
    return bboxes_nms


# Function to convert cells to bounding boxes
def convert_cells_to_bboxes(input_boxes, anchors, size: float, is_predictions=True) -> torch.Tensor:

    """
    inputs:
        - predictions: Tensor of shape (batch_size, grid_size, grid_size, anchors, 5 + num_classes)
        - anchors: List of anchors
        - size: Size of the image
        - is_predictions: Boolean value indicating whether the input is predictions or not

    returns:
        - bboxes: Tensor of shape (batch_size, grid_size * grid_size * anchors, 6)
        last channel is:
        (confidence(scores), x, y, width, height, best_class)
    """
    # Batch size used on predictions
    batch_size = input_boxes.shape[0]
    # Number of anchors
    num_anchors = len(anchors)
    # List of all the predictions
    box_predictions = input_boxes[..., 1:5] # x, y, w, h

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) # x, y
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors # w, h
        scores = torch.sigmoid(input_boxes[..., 0:1])
        best_class = torch.argmax(input_boxes[..., 5:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = input_boxes[..., 0:1]
        best_class = input_boxes[..., 5:6]


    # создаем матрицу индексов ячеек
    cell_indices_X = (torch.arange(size).view(1, 1, size, 1, 1).expand(batch_size, num_anchors, size, size, 1).to(DEVICE))
    cell_indices_Y = (torch.arange(size).view(1, 1, 1, size, 1).expand(batch_size, num_anchors, size, size, 1).to(DEVICE))


    # Calculate x, y, width and height with proper scaling
    x = 1 / size * (box_predictions[..., 0:1] + cell_indices_X)
    y = 1 / size * (box_predictions[..., 1:2] + cell_indices_Y)
    width_height = 1 / size * box_predictions[..., 2:4]

    # Concatinating the values and reshaping them in
    # (BATCH_SIZE, num_anchors * S * S, 6) shape
    converted_bboxes = torch.cat(
        (scores, x, y, width_height, best_class), dim=-1
    ).reshape(batch_size, num_anchors * size * size, 6)

    # Returning the reshaped and converted bounding box list
    return converted_bboxes


def get_true_bboxes_in_batch(
    input_bboxes,   
    iou_threshold,
    scaled_anchors,
    conf_threshold,
    is_preds=True
):
    """
    input_bboxes - Это или предсказания модели или label
    если это label:
    input_bboxes - list of tensors, each tensor is a batch of labels:
        [bs, num_anchors, grid_size, grid_size, 6]
        last 6 values are [is_object, x, y, w, h, class]
    если это предсказания:
    input_bboxes - list of tensors, each tensor is a batch of outputs:
        [bs, num_anchors, grid_size, grid_size, 25]
        last 25 values are [conf, x, y, w, h, 20 classes confidence]
    threshold - threshold for objectness score
    
    """

    all_boxes_array = np.array([], dtype=object)
    
    batch_size = input_bboxes[0].shape[0]
    num_shapes = len(input_bboxes)

    all_bboxes = torch.Tensor()

    for i in range(num_shapes):
        size = input_bboxes[i].shape[2]
        
        
        converted_boxes = convert_cells_to_bboxes(
            input_bboxes[i], anchors=scaled_anchors[i], size=size, is_predictions=is_preds
        )
        all_bboxes = torch.cat((all_bboxes, converted_boxes), dim=1)
    print(f"{all_bboxes.shape=}")

    all_bboxes[..., 1:5] = xywh2xyxy(all_bboxes[..., 1:5])

    boxes = all_bboxes[..., 1:5]
    scores = all_bboxes[..., 0]
    class_ids = all_bboxes[..., 5:]
    
    keep_indices = ops.batched_nms(boxes=boxes, scores=scores, idxs=class_ids, iou_threshold=iou_threshold)
    filtered_all_boxes = all_bboxes[keep_indices]
    print(f"{filtered_all_boxes.shape=}")



def get_true_bboxes(
    input_bboxes: list,   
    iou_threshold: float,
    scaled_anchors: torch.Tensor,
    conf_threshold: float,
    is_preds=True
) -> list:
    """
    inputs
        input_bboxes - Это или предсказания модели или label
        если это label:
        input_bboxes - list of tensors, each tensor is a batch of labels:
            [bs, num_anchors, grid_size, grid_size, 6]
            last 6 values are [object_confidence, x, y, w, h, class]
        если это предсказания:
        input_bboxes - list of tensors, each tensor is a batch of outputs:
            [bs, num_anchors, grid_size, grid_size, 25]
            last 25 values are [conf, x, y, w, h, 20 classes confidence]
        threshold - threshold for objectness score
    returns:
        list of tensors, each tensor is a batch of true outputs:
            [num_boxes, 6]
            last 6 values are [object_confidence, x, y, x, y, best_class]
    """

    all_boxes_list = []
    
    batch_size = input_bboxes[0].shape[0]
    num_shapes = len(input_bboxes)

    all_bboxes = torch.Tensor().to(DEVICE)

    for i in range(num_shapes):
        size = input_bboxes[i].shape[2]        
        
        converted_boxes = convert_cells_to_bboxes(
            input_bboxes[i], anchors=scaled_anchors[i], size=size, is_predictions=is_preds
        )
        all_bboxes = torch.cat((all_bboxes, converted_boxes), dim=1)
    

    all_bboxes[..., 1:5] = xywh2xyxy(all_bboxes[..., 1:5])


    
    for i in range(batch_size):
        one_img_boxes = all_bboxes[i].detach().to()
        # Фильтрация по confidence
        conf_mask = one_img_boxes[..., 0] >= conf_threshold
        one_img_boxes = one_img_boxes[conf_mask]

        img_boxes = one_img_boxes[..., 1:5]
        img_scores = one_img_boxes[..., 0]
        class_ids = one_img_boxes[..., 5:]
        
        keep_indices = ops.nms(boxes=img_boxes, scores=img_scores, iou_threshold=iou_threshold)
    
        filtered_img_boxes = one_img_boxes[keep_indices]
        all_boxes_list.append(filtered_img_boxes)

    return all_boxes_list


def save_checkpoint(model, optimizer, model_epoch, dataset_type:str, filename="my_checkpoint.pt", ):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": model_epoch,
        "model_dataset_type": dataset_type
    }
    torch.save(checkpoint, filename)


def convert_pascal_to_coco_model(checkpoint, model, optimizer, lr, model_epoch_list):
    """
    Задача функции - загрузить сохраннную модель которая обучалась на датасете pascal_voc,
    а потом заменить в ней выхходные слои для обучения на датасете coco

    # создаем модель, которая соответствуем сохраненной модели pascal_voc
    # создаем оптимизатор, который соответствуем сохраненному оптимизатору
    # загружаем модель и оптимизатор из чекпоинта
    # меняем выходные слои
    args:
        model - модель, которую мы загружаем
        optimizer - оптимизатор, который мы загружаем
        lr - learning rate, который мы загружаем
        model_epoch_list - список, в который мы добавляем номер эпохи, которую мы загружаем
    outputs:
        model, optimizer - модель и оптимизатор, которые мы загрузили и доработали
    """

    print("=> Starting model conversion from PASCAL VOC to COCO dataset")

    
    model = YOLO(num_classes=20)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model_epoch_list.append(checkpoint["epoch"])
    # Замеям выходные слои модели
    # head_1
    head1_pred_in_channels = model.head1.pred_layer.in_channels
    model.head1.pred_layer = nn.Conv2d(head1_pred_in_channels, (NUM_CLASSES + 5) * 3, kernel_size=1)
    model.head1.num_classes = NUM_CLASSES
    # head_2
    head2_pred_in_channels = model.head2.pred_layer.in_channels
    model.head2.pred_layer = nn.Conv2d(head2_pred_in_channels, (NUM_CLASSES + 5) * 3, kernel_size=1)
    model.head2.num_classes = NUM_CLASSES
    # head_3
    head3_pred_in_channels = model.head3.pred_layer.in_channels
    model.head3.pred_layer = nn.Conv2d(head3_pred_in_channels, (NUM_CLASSES + 5) * 3, kernel_size=1)
    model.head3.num_classes = NUM_CLASSES
        

    # Перемещаем состояние оптимизатора на нужное устройство
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    if NEED_TO_CHANGE_LR:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    optimizer = Adam([
            {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},  # Lower lr for pretrained layers
            {'params': model.head1.pred_layer.parameters(), 'lr': LEARNING_RATE},  # Higher lr for new layers
            {'params': model.head2.pred_layer.parameters(), 'lr': LEARNING_RATE},
            {'params': model.head3.pred_layer.parameters(), 'lr': LEARNING_RATE}
        ])


    return model, optimizer



def convert_coco_to_pascal_model(checkpoint, model, optimizer, lr, model_epoch_list):
    """
    Задача функции - загрузить сохраннную модель которая обучалась на датасете COCO,
    а потом заменить в ней выхходные слои для обучения на датасете PASCAL_VOC

    # создаем модель, которая соответствуем сохраненной модели COCO
    # создаем оптимизатор, который соответствуем сохраненному оптимизатору
    # загружаем модель и оптимизатор из чекпоинта
    # меняем выходные слои
    args:
        model - модель, которую мы загружаем
        optimizer - оптимизатор, который мы загружаем
        lr - learning rate, который мы загружаем
        model_epoch_list - список, в который мы добавляем номер эпохи, которую мы загружаем
    outputs:
        model, optimizer - модель и оптимизатор, которые мы загрузили и доработали
    """

    print("=> Starting model conversion from COCO to PASCAL VOC dataset")

    
    model = YOLO(num_classes=80)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model_epoch_list.append(checkpoint["epoch"])
    # Замеям выходные слои модели
    # head_1
    head1_pred_in_channels = model.head1.pred_layer.in_channels
    model.head1.pred_layer = nn.Conv2d(head1_pred_in_channels, (NUM_CLASSES + 5) * 3, kernel_size=1)
    model.head1.num_classes = NUM_CLASSES
    # head_2
    head2_pred_in_channels = model.head2.pred_layer.in_channels
    model.head2.pred_layer = nn.Conv2d(head2_pred_in_channels, (NUM_CLASSES + 5) * 3, kernel_size=1)
    model.head2.num_classes = NUM_CLASSES
    # head_3
    head3_pred_in_channels = model.head3.pred_layer.in_channels
    model.head3.pred_layer = nn.Conv2d(head3_pred_in_channels, (NUM_CLASSES + 5) * 3, kernel_size=1)
    model.head3.num_classes = NUM_CLASSES
        

    # Перемещаем состояние оптимизатора на нужное устройство
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    if NEED_TO_CHANGE_LR:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    optimizer = Adam([
            {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},  # Lower lr for pretrained layers
            {'params': model.head1.pred_layer.parameters(), 'lr': LEARNING_RATE},  # Higher lr for new layers
            {'params': model.head2.pred_layer.parameters(), 'lr': LEARNING_RATE},
            {'params': model.head3.pred_layer.parameters(), 'lr': LEARNING_RATE}
        ])

    
    return model, optimizer
    

def load_pretrained_model(checkpoint_path: str, model, optimizer, lr, model_epoch_list):
    """
    Если модель загружается из предобученной, то нужно понять является ли модель для coco или pascal
    """

    checkpoint = torch.load(checkpoint_path)
    # print("model_dataset_type: ", checkpoint["model_dataset_type"])
    # print("DATASET_TYPE: ", DATASET_TYPE)
    # Загрузка предобученной модели
    # try:
    # Если тип датасета и тип модели совпадают, то просто загружаем модель
    if DATASET_TYPE == checkpoint['model_dataset_type']:
        print(f"=> Загружаем модель из файла {checkpoint_path}")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        model_epoch_list.append(checkpoint["epoch"])
        model_dataset_type = checkpoint["model_dataset_type"]
    # Если тип датасета и тип модели не совпадают, то:

    # Если тип модели - PASCAL_VOC, а тип датасета - COCO, то:
    elif checkpoint["model_dataset_type"] == 'PASCAL_VOC':
        model, optimizer = convert_pascal_to_coco_model(checkpoint=checkpoint, model=model, optimizer=optimizer, lr=lr, model_epoch_list=model_epoch_list)
        print(f"=> Модель загружена из файла {checkpoint_path} обученную на датасете PASCAL_VOC. В модели заменены выходные слои для обучения на датасете COCO")
    # Если тип модели - COCO, а тип датасета - PASCAL_VOC, то:
    elif checkpoint["model_dataset_type"] == 'COCO':
        model, optimizer = convert_coco_to_pascal_model(checkpoint=checkpoint, model=model, optimizer=optimizer, lr=lr, model_epoch_list=model_epoch_list)
        print(f"=> Модель загружена из файла {checkpoint_path} обученную на датасете COCO. В модели заменены выходные слои для обучения на датасете PASCAL_VOC")
            
    # except:
    #     print("Не удалось загрузить модель")

    return model, optimizer



def xywh2xyxy(bboxes: torch.Tensor, box_format: str = "midpoint") -> torch.Tensor:
    """
    Convert bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2) format.
    """
    if box_format == "midpoint":
        box_x1 = bboxes[..., 0:1] - bboxes[..., 2:3] / 2
        box_y1 = bboxes[..., 1:2] - bboxes[..., 3:4] / 2
        box_x2 = bboxes[..., 0:1] + bboxes[..., 2:3] / 2
        box_y2 = bboxes[..., 1:2] + bboxes[..., 3:4] / 2

    box_xyxy = torch.cat([box_x1, box_y1, box_x2, box_y2], dim=-1)

    return box_xyxy

