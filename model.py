import torch
from torch import nn
import warnings
from model_architecture import model_architeture
warnings.filterwarnings('ignore')

class YOLO(nn.Module):

    def __init__(self, S, C, B, IMG_SIZE, architecture_config):

        super(YOLO, self).__init__()
        self.S = S
        self.C = C
        self.B = B
        self.IMG_SIZE = IMG_SIZE
        self.architecture_config = architecture_config

        backbone_blocks = []
        for block in self.architecture_config:
            backbone_blocks.append(self.create_block(block))

        backbone_blocks.append(nn.Flatten())
        backbone = nn.Sequential(*backbone_blocks)

        input_test = torch.rand((1, 3, self.IMG_SIZE, self.IMG_SIZE))
        output = backbone(input_test)
        in_features_linear = output.shape[-1]

        self.net = nn.Sequential(
            backbone, 
            nn.Linear(in_features=in_features_linear, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=(self.S * self.S * (1 + 4 + self.C)))
        )
    
    def create_block(self, block):
        
        block_type = block.__class__.__name__
        if (block_type == 'list'): # é uma camada convolucional
            in_channels, out_channels, kernel_size, stride, padding = block
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        elif (block_type == 'str'): # pode ser leaky ou maxpool
            layer_type, value = block.split(':')
            if (layer_type == 'maxpool'):
                return nn.MaxPool2d(kernel_size=int(value))
            elif (layer_type == 'leaky'):
                return nn.LeakyReLU(negative_slope=float(value))
            else:
                raise "Nome de camada não reconhecida! Avaliar a estrutura novamente."
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':

    ma = model_architeture()
    model = YOLO(4, 2, 1, 300, ma.architecture_config)
    input = torch.rand((2, 3, 300, 300))
    output = model(input)
    print (f'output.shape: {output.shape}')