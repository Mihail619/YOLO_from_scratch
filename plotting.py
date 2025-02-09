import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import torch
from typing import List

from config import PASCAL_CLASSES, COCO_CLASSES, DATASET_TYPE, PATH, ANCHORS, GRID_SIZES, DEVICE
from utils import xywh2xyxy, custom_nms, get_true_bboxes



if DATASET_TYPE == 'PASCAL_VOC':
    class_labels = PASCAL_CLASSES
else:
    class_labels = COCO_CLASSES

# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes: List[torch.Tensor], ax, num_images): 

    """
    input:
        image - image to plot
        list[torch.Tensor[num_boxes, [object_confidence, x, y, x, y, best_class]]]
    """
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
    
    # Reading the image with OpenCV 
    img = np.array(image) 
    
    # Переводим в формат ()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    elif img.shape[1] == 3:
        img = img.transpose(2, 0, 1)

    
    # Getting the height and width of the image 
    h, w, _ = img.shape 
    
    # Add image to plot 
    ax.imshow(img) 
  
    # Plotting the bounding boxes and labels over the image 
    for i, box in enumerate(boxes): 
        # Get the class from the box 
        class_pred = box[5]
        # Get the score from the box 
        score = box[0]
        # Get the box coordinates 
        box = box[1:5] 

        # Get the upper left corner coordinates 
        upper_left_x = box[0]
        upper_left_y = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        
        # Create a Rectangle patch with the bounding box 
        rect = patches.Rectangle( 
            (upper_left_x * w, upper_left_y * h), 
            width * w, 
            height * h, 
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
          
        # Add the patch to the Axes 
        ax.add_patch(rect) 
          
        # Add class name to the patch 
        ax.text( 
            upper_left_x * w, 
            upper_left_y * h, 
            s=f"{class_labels[int(class_pred)]} ({score:.2f})", 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        ) 
        
        if i >= num_images:
            break
  
    


@torch.inference_mode()
def plot_predicted_imgs(loader, model, confidence_threshold=0.5, iou_threshold=0.5, num_images=5):
    
    scaled_anchors = (
        torch.tensor(ANCHORS) / (
        1 / torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
    )
                     ).to(DEVICE)

    # Getting a batch from the dataloader 
    x, y = next(iter(loader))
    
    x = x.to(DEVICE)
    
    
    model.eval()
    output = model(x)
    
    # Получаем список для каждого изображения результирующих bboxe в формате
    # list[torch.Tensor[num_boxes, [object_confidence, x, y, x, y, best_class]]]

    bboxes_list = get_true_bboxes(input_bboxes=output, iou_threshold=iou_threshold, scaled_anchors=scaled_anchors, conf_threshold=confidence_threshold, is_preds=True)
    
    rows = int(np.ceil(num_images / 3))
    cols = min(num_images, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols

        if idx < len(bboxes_list):
            plot_image(x[idx].to("cpu"), bboxes_list[idx], ax=axes[row, col], num_images=num_images)

        else:
            axes[row, col].axis("off")
    
    # Display the plot 
    plt.tight_layout()
    plt.show()


def plot_ground_truth_imgs(loader, num_images=5):
    scaled_anchors = (
        torch.tensor(ANCHORS) / (
        1 / torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )).to(DEVICE)

    # Getting a batch from the dataloader 
    x, y = next(iter(loader))
    
    # Get bounding boxes for each image in batch
    bboxes_list = get_true_bboxes(
        input_bboxes=y,
        iou_threshold=0.9, # High threshold since these are ground truth boxes
        scaled_anchors=scaled_anchors,
        conf_threshold=0.9, # High threshold since these are ground truth boxes 
        is_preds=False
    )

    rows = int(np.ceil(num_images / 3))
    cols = min(num_images, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each image with its ground truth boxes
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        if idx < len(bboxes_list):
            plot_image(x[idx].to("cpu"), bboxes_list[idx], ax=axes[row, col], num_images=num_images)

        else:
            axes[row, col].axis("off")
        
    # Display the plot 
    plt.tight_layout()
    plt.show()