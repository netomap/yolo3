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

        blocks = self.create_blocks()
        self.net = nn.Sequential(*blocks)

    def create_blocks(self):
        
        backbone_blocks = []
        camada_apos_flatten = False
        in_features_depois_flatten = -1

        for block in self.architecture_config: # para cada bloco da arquitetura

            block_type = block.__class__.__name__ # pega o nome do tipo do bloco. Lista ou string

            if (block_type == 'list'): # é uma lista (pode ser conv ou linear)
                if (len(block) == 5): # é uma camada convolucional
                    in_channels, out_channels, kernel_size, stride, padding = block
                    backbone_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)) # adiciona convolucional
                elif (len(block) == 2): # é linear
                    in_features, out_features = block  # só tem dois elementos
                    if (camada_apos_flatten):  # avisa se essa layer é justamente após a camada flatten, para calcular o número de neurônios da entrada
                        in_features = in_features_depois_flatten
                        camada_apos_flatten = False # a partir dessa camada, já não é mais após a flatten. 
                    
                    if (out_features == -1):   # como padrão, inserir -1 nos campos que devem ser calculados, como neste caso que é a saída de 
                        out_features = self.S * self.S * (1 + 4 + self.C)  # neurônios da rede YOLO.

                    backbone_blocks.append(nn.Linear(in_features, out_features))  # adiciona os valores calculados e/ou digitados.
                else:
                    raise "lista com valores diferente de 5 (conv) ou 2 (linear)"

            elif (block_type == 'str'): # pode ser leaky ou maxpool
                layer_type, value = block.split(':')
                if (layer_type == 'maxpool'):
                    backbone_blocks.append(nn.MaxPool2d(kernel_size=int(value))) # adiciona maxpool
                elif (layer_type == 'leaky'):
                    backbone_blocks.append(nn.LeakyReLU(negative_slope=float(value))) # adiciona leakyRELU
                elif (layer_type == 'flatten'):
                    backbone_blocks.append(nn.Flatten())
                    model_test = nn.Sequential(*backbone_blocks)                    # cria um modelo teste rapidamente
                    input_test = torch.rand((1, 3, self.IMG_SIZE, self.IMG_SIZE))   # simula uma entrada simples
                    output = model_test(input_test)                                 # calcula a saída
                    in_features_depois_flatten = output.shape[-1]                   # pega o shape da saída flatten para servir de entrada nas camadas lineares
                    camada_apos_flatten = True      # marca que essa camada é flatten e a próxima é True
                else:
                    raise "Nome de camada não reconhecida! Avaliar a estrutura novamente."
        
        return backbone_blocks
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':

    ma = model_architeture()
    model = YOLO(4, 2, 1, 300, ma.architecture_config)
    input_test = torch.rand((1, 3, 300, 300))
    output = model(input_test)
    print (model)
    print (output.shape)