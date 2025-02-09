import sys
import torch
import torchmetrics
from typing import List

from torchvision.ops import nms
from pprint import pprint

from config import DEVICE, ANCHORS, NUM_CLASSES
from utils import xywh2xyxy

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, task='binary', num_classes=None) -> tuple[float, float, float]:
    # Создаем объекты метрик
    accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
    precision = torchmetrics.Precision(task=task, num_classes=num_classes)
    recall = torchmetrics.Recall(task=task, num_classes=num_classes)
    
    # Вычисляем метрики
    acc = accuracy(predictions, targets)
    prec = precision(predictions, targets)
    rec = recall(predictions, targets)
    
    
    return {"acc":acc, "prec":prec, "rec":rec}


def check_metrics(labels: torch.Tensor, outputs:torch.Tensor, confidence_threshold:float):
    '''
    inputs
        y - list of tensors, each tensor is a batch of labels:
            [bs, num_anchors, grid_size, grid_size, 6]
            last 6 values are [is_object, x, y, w, h, class]
        output - list of tensors, each tensor is a batch of outputs:
            [bs, num_anchors, grid_size, grid_size, 25]
            last 25 values are [conf, x, y, w, h, 20 classes confidence]
        confidence_threshold - threshold for objectness score

    returns: 
        (all_class_metrics = {"acc":torch.Tensor, "prec":torch.Tensor, "rec":torch.Tensor},
         all_obj_metrics = {"acc":torch.Tensor, "prec":torch.Tensor, "rec":torch.Tensor},
         all_noobj_metrics = {"acc":torch.Tensor, "prec":torch.Tensor, "rec":torch.Tensor})
    '''
    metrics = {
        'class': {'acc': [], 'prec': [], 'rec': []},
        'obj': {'acc': [], 'prec': [], 'rec': []},
        'noobj': {'acc': [], 'prec': [], 'rec': []}
    }
    
    num_shapes = len(labels)

    for i in range(num_shapes):
        scale_label = labels[i].to(DEVICE)
        scale_output = outputs[i]

        obj_mask = labels[i][..., 0] == 1 
        noobj_mask = labels[i][..., 0] == 0

        # выведеем данные для предсказаний
        pred_class = torch.argmax(scale_output[..., 5:][obj_mask], dim=-1)
        pred_confidence = torch.sigmoid(scale_output[..., 0]) > confidence_threshold
        pred_obj = pred_confidence[obj_mask]
        pred_noobj = pred_confidence[noobj_mask]
              


        label_class = scale_label[..., 5][obj_mask]
        label_obj = scale_label[..., 0][obj_mask]
        label_noobj = scale_label[..., 0][noobj_mask]
        
        # Вычислим метрики 
        # результат - tuple[accuracy, precision, recall]
        class_metrics = calculate_metrics(pred_class, label_class, task='multiclass', num_classes=NUM_CLASSES)
        obj_metrics = calculate_metrics(pred_obj, label_obj, task='binary')
        noobj_metrics = calculate_metrics(pred_noobj, label_noobj, task='binary')

        
        for key in metrics['class'].keys():
            metrics['class'][key].append(class_metrics[key])
            metrics['obj'][key].append(obj_metrics[key])
            metrics['noobj'][key].append(noobj_metrics[key])
        

    
    # Average metrics
    for metric_type in metrics: # class, obj, noobj
        for key in metrics[metric_type]: # acc, prec, rec
            metrics[metric_type][key] = torch.stack(metrics[metric_type][key]).mean().cpu() # mean acc in class

    
    
    return metrics
    