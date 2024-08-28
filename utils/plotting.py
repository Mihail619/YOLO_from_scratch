import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List
import time
import tqdm
import logging


from config import (COCO_CLASSES, PASCAL_CLASSES, DATASET, DEVICE, ANCHORS, SIZES)
from utils import intersection_over_union


def nms_for_plotting(bboxes, iou_threshold, threshold, box_format="corners")-> List[list]:
    """
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    assert type(bboxes) == list
    bboxes = np.array(bboxes)

    # Фильтрация по порогу уверенности
    mask = bboxes[:, 1] > threshold
    bboxes = bboxes[mask]

    # сортировка по уверенности
    order = bboxes[:, 1].argsort()[::-1]
    bboxes = bboxes[order]

    keep = []
    while bboxes.shape[0] > 0:
        keep.append(bboxes[0])
        if bboxes.shape[0] == 1:
            break
        # Вычисление IoU векторизованно
        ious = compute_iou_numpy(bboxes[0, 2:], bboxes[1:, 2:], box_format)
    
        # Фильтрация боксов
        mask = (ious < iou_threshold) | (bboxes[1:, 0] != bboxes[0, 0])
        bboxes = bboxes[1:][mask]
    
    return keep


def compute_iou_numpy(box, boxes, box_format="corners"):
    if box_format == "corners":
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        left_top = np.maximum(box[:2], boxes[:, :2])
        right_bottom = np.minimum(box[2:], boxes[:, 2:])
        
        wh = np.maximum(right_bottom - left_top, 0)
        inter_area = wh[:, 0] * wh[:, 1]
        
        union_area = box_area + boxes_area - inter_area
        iou = inter_area / union_area
    
    elif box_format == "midpoint":
        box_x1, box_y1 = box[0] - box[2]/2, box[1] - box[3]/2
        box_x2, box_y2 = box[0] + box[2]/2, box[1] + box[3]/2
        
        boxes_x1 = boxes[:, 0] - boxes[:, 2]/2
        boxes_y1 = boxes[:, 1] - boxes[:, 3]/2
        boxes_x2 = boxes[:, 0] + boxes[:, 2]/2
        boxes_y2 = boxes[:, 1] + boxes[:, 3]/2
        
        left_top = np.maximum([box_x1, box_y1], np.column_stack([boxes_x1, boxes_y1]))
        right_bottom = np.minimum([box_x2, box_y2], np.column_stack([boxes_x2, boxes_y2]))
        
        wh = np.maximum(right_bottom - left_top, 0)
        inter_area = wh[:, 0] * wh[:, 1]
        
        box_area = box[2] * box[3]
        boxes_area = boxes[:, 2] * boxes[:, 3]
        
        union_area = box_area + boxes_area - inter_area
        iou = inter_area / union_area
    
    else:
        raise ValueError("Unsupported box format. Use 'corners' or 'midpoint'.")
    
    return iou


def cells_to_bboxes(predictions: torch.Tensor, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (BS, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""

    cmap = plt.get_cmap("tab20b")
    class_labels = COCO_CLASSES if DATASET=='COCO' else PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()



def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()

    x, y = next(iter(loader))
    x = x.to(DEVICE)
    bs = x.shape[0]

    with torch.no_grad():
        # вычисляется выход модели [(), (), ()]
        out = model(x)
        
        bboxes = [[] for _ in range(bs)]
        for i in range(3):
            print("output_shape: ", out[i].shape)
            # 
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i] # Список bboxes на данном масштабе'

            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    print(f"bboxes: {len(bboxes)} | bboxes[0]: {len(bboxes[0])} | BBOXES[0][0]:  {bboxes[0][0]}")


    for i in range(bs):
        print("Вычисление nms...")
        start_time = time.time()
        nms_boxes = nms_for_plotting(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"NMS execution time: {execution_time:.4f} seconds")
        print("Отрисовка изображения")
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)
        break


if __name__ == "__main__":
    import matplotlib.patches as patches
    from utils import get_data_loaders
    import os
    from config import PATH
    # create loaders
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ALBUMENTATIONS_DISABLE_CHECKING_VERSION'] = '1'


    train_loader, valid_loader = get_data_loaders(path=PATH, num_images=2)

    
    #_______________________________________________________________________________________________

    from my_yolo_model import YOLO
    from config import NUM_CLASSES, NUMBER_BLOCKS_LIST, BACKBONE_NUM_CHANNELS
    # check how to work non max suppression
    model = YOLO(num_classes=NUM_CLASSES, n_blocks_list=NUMBER_BLOCKS_LIST, backbone_num_channels=BACKBONE_NUM_CHANNELS)

    model.eval() 
    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(DEVICE)
    plot_couple_examples(model, train_loader, thresh=0.5, iou_thresh=0.5, anchors=scaled_anchors)
