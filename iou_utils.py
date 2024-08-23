import torch

# Defining a function to calculate Intersection over Union (IoU) 
def iou(box1: torch.Tensor, box2: torch.Tensor, is_pred=True): 
	 
    # IoU score based on width and height of bounding boxes 
    
    # Calculate intersection area 
    intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 

    # Calculate union area 
    box1_area = box1[..., 0] * box1[..., 1] 
    box2_area = box2[..., 0] * box2[..., 1] 
    union_area = box1_area + box2_area - intersection_area 

    # Calculate IoU score 
    iou_score = intersection_area / union_area 

    # Return IoU score 
    return iou_score