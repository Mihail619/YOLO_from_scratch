import torch 
from torch import nn 
from torchvision.ops import FeaturePyramidNetwork
from typing import List, Dict

from config import ANCHORS

num_anchors = len(ANCHORS[0])


class Conv_3x3_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, stride: int=1):
        super(Conv_3x3_block, self).__init__()
        
        self.layer_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.activation = activation()
        self.norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.norm(out)
        out = self.activation(out)
        
        return(out)


class CSPBlock_Residual(nn.Module):
    def __init__(self, n_channels, block: Conv_3x3_block, activation, n_blocks=1):
        super(CSPBlock_Residual, self).__init__()
        
        self.layers = nn.ModuleList([block(in_channels=n_channels//2, out_channels=n_channels//2, activation=activation) for i in range(n_blocks)])
        self.n_channels = n_channels
        # print("CSP n_blocks = ", n_blocks)
        # print("layers", len(self.layers))

    def forward(self, x):
        
        # print("CSP IN :", x.shape)
        
        split_torch = torch.chunk(x, chunks=2, dim=1)
        
        residual, out = split_torch[0], split_torch[1]
        
        # print("CSP residual, out", residual.shape, out.shape)
        
        for block in self.layers:
            out = block(out) + out
            
        out = torch.cat((residual, out), dim=1)
        # print("CSP_OUT: "out.shape)
        
        return out
        

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, norm_activation=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias= not norm_activation, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation  = nn.LeakyReLU(0.1)
        self.use_bn_activation = norm_activation

    def forward(self, x):
        if self.use_bn_activation:
            return self.activation(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_blocks=1, use_residual=True, block=CNNBlock):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers += [nn.Sequential(
                block(in_channels=channels, out_channels=channels//2, kernel_size=1),
                block(in_channels=channels//2, out_channels=channels//2, kernel_size=3, padding=1),
                block(in_channels=channels//2, out_channels=channels , kernel_size=1)
                )
            ]
        
        self.use_residual = use_residual
        self.num_blocks = num_blocks

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x) if self.use_residual else layer(x)
            else:
                x = layer(x)
        return x 
    

class Head(nn.Module):
    """
    input: Tensor[bs, ]
    Returns a tensor of shape [B, num_ancors, H, W, 3 * num_classes + 5]"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # переименовать как у меня в голове
        self.head = nn.Sequential(
            CNNBlock(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, padding=1),
            CNNBlock(in_channels=in_channels*2, out_channels= 3 * (num_classes + 5), kernel_size=1, norm_activation=False)
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x).reshape(x.shape[0], num_anchors, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
    
        return x
    

class DarkNet53(nn.Module):
    def __init__(self, out_channels_list: List[torch.Tensor], n_blocks_list: List[int], block=CSPBlock_Residual, actvation=nn.ReLU):
        super().__init__()
        self.out_channels_list = out_channels_list
        self.n_blocks_list = n_blocks_list

        in_chan, out_chan = 3, 64
        self.layer_1 = Conv_3x3_block(in_channels=3, out_channels=out_chan, activation=actvation, stride=2)
        in_chan, out_chan = out_chan, 128

        self.layer_2 = Conv_3x3_block(in_channels=in_chan, out_channels=out_chan, activation=actvation, stride=2)
        #in_chan, out_chan = out_chan, out_chan
        self.layer_3 = block(n_channels=out_chan, block=Conv_3x3_block, activation=actvation, n_blocks=n_blocks_list[0])

        in_chan, out_chan = out_chan, out_channels_list[0]
        self.layer_4 = Conv_3x3_block(in_channels=in_chan, out_channels=out_chan, activation=actvation, stride=2)
        # in_chan, out_chan = out_chan, out_chan
        self.layer_5 = block(n_channels=out_chan, block=Conv_3x3_block, activation=actvation, n_blocks=n_blocks_list[1])

        in_chan, out_chan = out_chan, out_channels_list[1]
        self.layer_6 = Conv_3x3_block(in_channels=in_chan, out_channels=out_chan, activation=actvation, stride=2)
        self.layer_7 = block(n_channels=out_chan, block=Conv_3x3_block, activation=actvation, n_blocks=n_blocks_list[2])

        in_chan, out_chan = out_chan, out_channels_list[2]
        self.layer_8 = Conv_3x3_block(in_channels=in_chan, out_channels=out_chan, activation=actvation, stride=2)
        self.layer_9 = block(n_channels=out_chan, block=Conv_3x3_block, activation=actvation, n_blocks=n_blocks_list[3])



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}
        x = self.layer_1(x)
        # print("layer_1: ", x.shape)
        x = self.layer_2(x)
        # print("layer_2: ", x.shape)
        x = self.layer_3(x)
        # print("layer_3: ", x.shape)
        x = self.layer_4(x)
        # print("layer_4: ", x.shape)
        x = self.layer_5(x)
        # print("layer_5: ", x.shape)
        output["Lowlevel_output"] = x

        x = self.layer_6(x)
        # print("layer_6: ", x.shape)
        x = self.layer_7(x)
        # print("layer_7: ", x.shape)
        output["Midlevel_output"] = x

        x = self.layer_8(x)
        # print("layer_8: ", x.shape)
        x = self.layer_9(x)
        # print("layer_9: ", x.shape)
        output["Highlevel_output"] = x

        return output






class YOLO(nn.Module):
    def __init__(self, num_classes: int, backbone_num_channels: list, n_blocks_list: list, in_channels=3,  block=CNNBlock, BackBone_block=DarkNet53, ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        backbone_num_channels = backbone_num_channels
        fpn_out_channels = 128
        

        self.backbone = BackBone_block(out_channels_list=backbone_num_channels, n_blocks_list=n_blocks_list, block=CSPBlock_Residual, actvation=nn.ReLU)
        self.fpn = FeaturePyramidNetwork(in_channels_list=backbone_num_channels, out_channels=fpn_out_channels)
        
        self.head_1 = Head(in_channels=fpn_out_channels, num_classes=num_classes)
        self.head_2 = Head(in_channels=fpn_out_channels, num_classes=num_classes)
        self.head_3 = Head(in_channels=fpn_out_channels, num_classes=num_classes)

        self.block = block
                
    def forward(self, x) -> List[torch.Tensor]:
        
        
        backbone_out = self.backbone(x)
        # print("Backbone result: ", [x.shape for x in backbone_out.values()])
        # outputs.append(x)
        fpn_out = self.fpn(backbone_out)
        # print("fpn_out keys:   ", [x for x in fpn_out.keys()])
        # print("fpn_out values: ", [x.shape for x in fpn_out.values()])
        out_1 = self.head_1(fpn_out["Lowlevel_output"])
        # print("Head_1: ", out_1.shape)
        out_2 = self.head_2(fpn_out["Midlevel_output"])
        # print("Head_2: ", out_2.shape)
        out_3 = self.head_3(fpn_out["Highlevel_output"])
        # print("Head_3: ", out_3.shape)

        return out_1, out_2, out_3
    


if __name__ == "__main__":
    from dataset import VOCYOLODataset
    from config import PATH, ANCHORS, SIZES, NUM_CLASSES, IMAGE_SIZE
    from utils import get_data_loaders
    
    num_images = 2
    train_loader, test_loader = get_data_loaders(PATH, num_images=num_images)
    for x, y in train_loader:
        print("Y: ", [y.shape for y in y])
        print("X: ", x.shape)
        print(100 * "-", '\n', "Инициализация модели")
        model = YOLO(num_classes=NUM_CLASSES, backbone_num_channels=[128, 256, 512],n_blocks_list=[3, 4, 4, 3])
        model.eval() 
        print(100 * "-", '\n', "Результат модели \n", 100 * "-")
        out = model(x)
        print(100 * "-", '\n', "Сравнение результатов \n", 100 * "-")
        
        assert model(x)[0].shape == (num_images, 3, IMAGE_SIZE//8,  IMAGE_SIZE//8,  NUM_CLASSES + 5)
        assert model(x)[1].shape == (num_images, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, NUM_CLASSES + 5)
        assert model(x)[2].shape == (num_images, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, NUM_CLASSES + 5)

        print(len(out), len(y))
        
        