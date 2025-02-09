import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.ciou_loss import complete_box_iou_loss
from torchvision.ops import sigmoid_focal_loss

from utils import iou




class YoloLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        # self.focal_loss = FocalLoss()
        self.num_classes = num_classes

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1


    def forward(self, predictions, targets, scaled_anchors: torch.Tensor):
        # predictions: tensor([..., 25]). - is_obj, x, y, w, h, (20 classes)
        # targets: [..., 6]. - is_obj, x, y, w, h, class
        # scale_anchors - list[]
        object_mask = targets[..., 0] == 1     # mask for objects
        no_object_mask = targets[..., 0] == 0     # mask for NO objects
        # print(f"{predictions[is_object_mask].shape=}")
        # NO OBJECT LOSS
        
        no_object_loss = self.bce((predictions[..., 0:1][no_object_mask]), (targets[..., 0:1][no_object_mask]))

        # OBJECT LOSS
        
        scaled_anchors = scaled_anchors.reshape(1, len(scaled_anchors), 1, 1, 2) # изменим размер тензора якорей для возможного умножению на результат модели
        box_preds = torch.cat((torch.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * scaled_anchors), dim =-1)
        
        ious = iou(box_preds[object_mask], targets[..., 1:5][object_mask]).detach()
        
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][object_mask]), ious * targets[..., 0:1][object_mask])

        # BOX LOSS

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        targets[..., 3:5] = torch.log(1e-16 + targets[..., 3:5] / scaled_anchors)  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][object_mask], targets[..., 1:5][object_mask])

        
        # CLASS LOSS
        
        class_loss = self.cross_entropy(predictions[..., 5:][object_mask], targets[..., 5][object_mask].long())
        
        
        print(f"{box_loss.item()=}")
        print(f"{object_loss.item()=}")
        print(f"{class_loss.item()=}")
        print(f"{no_object_loss.item()=}")
        
        return  (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
    




if __name__ == "__main__":
    from config import PATH, ANCHORS, SIZES
    
    num_imgs = 10
    num_classes = 20
    
    # target: [img, [torch.Tensor([3, 13, 13, 6] ), torch.Tensor([3, 52, 52, 6]), torch.Tensor([3, 52, 52, 6]) ] ]
    import os
    from config import PATH, DEVICE
    from dataloader import get_data_loaders
    # create loaders
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ALBUMENTATIONS_DISABLE_CHECKING_VERSION'] = '1'


    train_loader, valid_loader = get_data_loaders(path=PATH, num_images=2)

    
    #_______________________________________________________________________________________________

    from my_yolo_model import YOLO
    from config import NUM_CLASSES, GRID_SIZES
    # check how to work non max suppression
    model = YOLO(num_classes=NUM_CLASSES, in_channels=3)

    model.eval() 

    # Scaling the anchors 
    scaled_anchors = ( 
        torch.tensor(ANCHORS) * 
        torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
        ).to(DEVICE) 

    for x, targets in train_loader:
        predictions = model(x)
        predictions_1, predictions_2, predictions_3 = predictions


        
        loss_fn = YoloLoss(num_classes=num_classes)
        print("Predictions: ", predictions[0].shape)
        print("Targets:     ", targets[0].shape)
        print(100*"_")
        
        image_labels = targets[0] 
        prediction_1 = predictions[0]
        anchor_1 = scaled_anchors[0]

        loss1 = loss_fn(predictions=predictions[0], targets=targets[0], scale_anchors=scaled_anchors[0])
        loss2 = loss_fn(predictions=predictions[1], targets=targets[1], scale_anchors=scaled_anchors[1])
        loss3 = loss_fn(predictions=predictions[2], targets=targets[2], scale_anchors=scaled_anchors[2])
        loss = loss1 + loss2 + loss3
        print("LOSS = ",loss)
        
        break

    # loss_fn(predictions, target, ANCHORS)


