import torch 
from torch import nn 


from config import ANCHORS

num_anchors = len(ANCHORS[0])

# Defining CNN Block 
class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__() 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 
		self.use_batch_norm = use_batch_norm 

	def forward(self, x): 
		# Applying convolution 
		x = self.conv(x) 
		# Applying BatchNorm and activation if needed 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x
		

# Defining residual block 
class ResidualBlock(nn.Module): 
	def __init__(self, channels, use_residual=True, num_repeats=1): 
		super().__init__() 
		
		# Defining all the layers in a list and adding them based on number of 
		# repeats mentioned in the design 
		res_layers = [] 
		for _ in range(num_repeats): 
			res_layers += [ 
				nn.Sequential( 
					nn.Conv2d(channels, channels // 2, kernel_size=1), 
					nn.BatchNorm2d(channels // 2), 
					nn.LeakyReLU(0.1), 
					nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
					nn.BatchNorm2d(channels), 
					nn.LeakyReLU(0.1) 
				) 
			] 
		self.layers = nn.ModuleList(res_layers) 
		self.use_residual = use_residual 
		self.num_repeats = num_repeats 
	
	# Defining forward pass 
	def forward(self, x): 
		for layer in self.layers: 
			residual = x 
			x = layer(x) 
			if self.use_residual: 
				x = x + residual 
		return x


# Defining scale prediction class 
class Head(nn.Module): 
	"""В предыдущем исполнении - этот класс назывался ScalePrediction"""
	def __init__(self, in_channels, num_classes): 
		super().__init__() 
		# Defining the layers in the network 
		self.features = nn.Sequential( 
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
			nn.BatchNorm2d(2*in_channels), 
			nn.LeakyReLU(0.1)			
		) 
		self.pred_layer = nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1)
		self.num_classes = num_classes 
	
	# Defining the forward pass and reshaping the output to the desired output 
	# format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
	def forward(self, x): 
		output = self.features(x) 
		output = self.pred_layer(output)
		output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3)) 
		output = output.permute(0, 1, 3, 4, 2) 
		return output


class Backbone(nn.Module):
	# Возвращет выход из каждого блока. От самой низкой размерности до самой высокой.
        # output_1 - 256, output_2 - 512  output_3 - 1024	
	
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1)
        )
        
        self.block1 = nn.Sequential(
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2)
        )
        
        self.block2 = nn.Sequential(
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8)
        )
        
        self.block3 = nn.Sequential(
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8)
        )
        
        self.block4 = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4)
        )

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2) #256 # First route connection
        x4 = self.block3(x3) #512 # Second route connection
        x5 = self.block4(x4) #1024
        return x3, x4, x5




class Neck(nn.Module):
    def __init__(self):
        # """ifif"""
        super().__init__()
        self.block_1 = nn.Sequential(
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
			CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
			ResidualBlock(1024, use_residual=False, num_repeats=1),
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
			
        )
		
        self.resize_1 = CNNBlock(512, 256, kernel_size=1, stride=1, padding=0)
        self.resize_2 = CNNBlock(256, 128, kernel_size=1, stride=1, padding=0) 
        
        # ===================================================
        self.block_2 = nn.Sequential(
            
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0), 
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(512, use_residual=False, num_repeats=1), 
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0), 
        )
        
        
        # ===================================================
        self.block_3 = nn.Sequential(
            
            
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0), 
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1), 
            ResidualBlock(256, use_residual=False, num_repeats=1), 
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0)
            )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, route_1, route_2, route_3):
        # input: route_1 = 256, route_2 = 512, route_3 = 1024
        # print("neck, route", route_1.shape, route_2.shape, route_3.shape)
        # =========
        #block_1
        #==========
        x = self.block_1(route_3) # возвращает 512
        output_1 = x.clone()
        # print("neck, block_1", x.shape)
        # =========
        #block_2
        #==========
        x = self.resize_1(x)    # возвращает 256
        x = self.upsample(x)    # увеличивает размер в 2 раза       
        # print("neck, block_2, upsample", x.shape)		
        x = torch.cat([x, route_2], dim=1) 
        # print("neck, block_2 after cat", x.shape)
        x = self.block_2(x) # возвращает 256
        output_2 = x.clone()
        # print("neck, block_2", x.shape)
        # =========
        #block_2
        #==========
        x = self.resize_2(x) # возвращает 128
        x = self.upsample(x) # увеличивает размер в 2 раза
        # print("neck, block_3, upsample", x.shape)
        x = torch.cat([x, route_1], dim=1)
        # print("neck, block_3 after cat", x.shape)
        output_3 = self.block_3(x) # возвращает 128
        # print("neck, block_3", x.shape)

        
        return output_1, output_2, output_3
		

        
	

class YOLO(nn.Module):
	def __init__(self, in_channels=3, num_classes=20):
		super().__init__()
		self.num_classes = num_classes 
		self.in_channels = in_channels

		self.backbone = Backbone(in_channels)
		self.neck = Neck()

		self.head1 = Head(512, num_classes=num_classes)

		self.head2 = Head(256, num_classes=num_classes)

		self.head3 = Head(128, num_classes=num_classes)

		self.upsample = nn.Upsample(scale_factor=2)

	def forward(self, x):
	
		route_1, route_2, route_3 = self.backbone(x) #256, 512, 1024

		neck_out_1, neck_out_2, neck_out_3 = self.neck(route_1, route_2, route_3) #512, 256, 

		head1_out = self.head1(neck_out_1)
		head2_out = self.head2(neck_out_2)
		head3_out = self.head3(neck_out_3)

		return [head1_out, head2_out, head3_out]
	

if __name__=="__main__":
    from utils import convert_pascal_to_coco_model
    from torch.optim import Adam


    CHECKPOINT_FILE = "models\\V1_E_  161_LOSS_34.86.pt"
    LEARNING_RATE = 1e-3
    DATASET_TYPE = "PASCAL_VOC"
    # Create sample input
    batch_size = 2
    channels = 3
    height = 416  # Standard YOLO input size
    width = 416
    num_classes = 20

    model = YOLO(in_channels=3, num_classes=20)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    model_epoch_list = []
	
    # print(model)
    new_state_dict = model.state_dict()
    keys_list = list(new_state_dict.keys())
    optimizer_new_state_dict = optimizer.state_dict()
    optimizer_new_state_dict_keys = list(optimizer_new_state_dict.keys())
    print(optimizer_new_state_dict)
	# Print all keys
    import pickle
    # with open('test/new_moelw_state_dict.pkl', 'wb') as f:
    #     pickle.dump(keys_list, f)
    # load_checkpoint(
    #             CHECKPOINT_FILE, model, optimizer, LEARNING_RATE, model_epoch_list, model_dataset_type=DATASET_TYPE
    #         )
    model.eval()
    x = torch.randn((batch_size, channels, height, width))
    outputs = model(x)

        # Test output shapes
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    # Expected shapes for each scale
    expected_shapes = [
        (batch_size, 3, height//32, width//32, num_classes + 5),  # Scale 1
        (batch_size, 3, height//16, width//16, num_classes + 5),  # Scale 2
        (batch_size, 3, height//8, width//8, num_classes + 5),    # Scale 3
    ]

    # Verify output shapes
    for idx, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == expected_shape, f"Scale {idx+1} shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("All tests passed successfully!")

