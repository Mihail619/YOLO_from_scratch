import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.ciou_loss import complete_box_iou_loss
from torchvision.ops import sigmoid_focal_loss



class YoloLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        # self.focal_loss = FocalLoss()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
        # predictions: tensor([..., 25]). - is_obj, x, y, w, h, (20 classes)
        # targets: [..., 6]. - is_obj, x, y, w, h, class
        is_object_mask = (targets[..., 0] == 1)     # mask for objects
        no_object_mask = (targets[..., 0] == 0)     # mask for NO objects

        class_preds = predictions[..., 5:]
        
        class_targets  = F.one_hot(targets[..., 5].long(), num_classes=self.num_classes).float()
        
        
        #____________________________________
        # NO OBJECT LOSS
        # ___________________________________
        no_object_loss = self.bce(predictions[..., 0:1][no_object_mask], targets[..., 0:1][no_object_mask])

        #____________________________________
        # OBJECT LOSS
        # ___________________________________
        obj_loss = self.bce(predictions[..., 0:1][is_object_mask], targets[..., 0:1][is_object_mask])

        #____________________________________
        # CLASS LOSS
        #____________________________________

        focal_loss = sigmoid_focal_loss(class_preds, class_targets, reduction='mean')
        
        #____________________________________
        # BOX LOSS
        #____________________________________

        box_loss = self.custom_ciou_loss(pred_boxes=predictions[..., 1:5], target_boxes=targets[..., 1:5], mask=is_object_mask)

        # print(box_loss)
        # print(obj_loss)
        # print(focal_loss)
        return obj_loss + focal_loss + box_loss + no_object_loss
    

    def custom_ciou_loss(sef, pred_boxes, target_boxes, mask):
        """ Реализация CIoU loss
        На вход bbox подаются в формате [x, y, w, h]. 
        Преобразовываются их в формат для https://pytorch.org/vision/main/_modules/torchvision/ops/ciou_loss.html 
        'Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``, and The two boxes should have the
        same dimensions.'
        Далее вычисляем CIOU только для тех объектов в которых есть объект. А именно в которых [...,0] == 1.
        Эти данные лежат в маске объектов mask состоящей из bool элементов. Ture - в тех, где есть объекты, False - в тех, где нет. 
        
        """
        # print(pred_boxes[..., 1:5])

        pred_x1y1 = pred_boxes[mask][..., 0:2] - pred_boxes[mask][..., 2:4] / 2
        pred_x2y2 = pred_boxes[mask][..., 0:2] + pred_boxes[mask][..., 2:4] / 2
        pred_boxes_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        target_x1y1 = target_boxes[mask][..., 0:2] - target_boxes[mask][..., 2:4] / 2
        target_x2y2 = target_boxes[mask][..., 0:2] + target_boxes[mask][..., 2:4] / 2
        target_boxes_xyxy = torch.cat([target_x1y1, target_x2y2], dim=-1)
   
        # ciou_loss = complete_intersection_over_union()
        loss = complete_box_iou_loss(pred_boxes_xyxy, target_boxes_xyxy) 
        loss = loss.mean()
        
        return loss



if __name__ == "__main__":
    from config import PATH, ANCHORS, SIZES
    from dataset import VOCYOLODataset
    num_imgs = 10
    num_classes = 20
    
    # target: [img, [torch.Tensor([3, 13, 13, 6] ), torch.Tensor([3, 52, 52, 6]), torch.Tensor([3, 52, 52, 6]) ] ]
    import os
    from config import PATH
    from utils import get_data_loaders
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
    for x, target in train_loader:
        predictions = model(x)
        predictions_1, predictions_2, predictions_3 = predictions


        
        loss_fn = YoloLoss(num_classes=num_classes)
        print("Predictions: ", predictions[0].shape)
        print("Targets:     ", target[0].shape)
        print(100*"_")
        
        image_labels = target[0]
        loss = loss_fn(predictions=predictions[0], targets=image_labels)
        print(loss)
        
        break

    # loss_fn(predictions, target, ANCHORS)


