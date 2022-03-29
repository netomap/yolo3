import torch
from torch import nn
from model import YOLO
from dataset import yolo_dataset, preparar_dataset_e_dataloaders
from utils import validacao, salvar_checkpoint, salvar_resultado_uma_epoca
from loss import YOLO_LOSS
from tqdm import tqdm
import numpy as np
import argparse
from tqdm import tqdm
from model_architecture import model_architeture

parser = argparse.ArgumentParser(description='Treinamento de uma rede neural YOLO.')
parser.add_argument('--s', type=int, default=4, help='Número de grids para divisão da imagem')
parser.add_argument('--b', type=int, default=1, help='Número de bouding boxes.')
parser.add_argument('--c', type=int, default=1, help='Número de classes do problema. Neste caso, somente uma.')
parser.add_argument('--imgsize', type=int, default=300, help='Tamanho a ser redimensionado cada imagem.')
parser.add_argument('--batchsize', type=int, default=16, help='Tamanho do lote para treinamento da rede.')
parser.add_argument('--testsize', type=float, default=0.1, help='Tamanho (em percentual) da divisão treino-teste para validação cruzada.')
parser.add_argument('--lr', type=float, default=1e-3, help='LEARNING RATE')
parser.add_argument('--e', type=int, default=10, help='Número de épocas para treinamento')
parser.add_argument('--lc', type=int, default=1, help='Taxa de penalização para a perda das coordenadas.')
parser.add_argument('--lo', type=int, default=1, help='Taxa de penalização para a perda quando encontra algum objeto.')
parser.add_argument('--lno', type=int, default=10, help='Taxa de penalização para a perda quando não encontra objeto.')

args = parser.parse_args()
print (args)

S = args.s
C = args.c
B = args.b
IMG_SIZE = args.imgsize
BATCH_SIZE = args.batchsize
TEST_SIZE = args.testsize
LEARNING_RATE = args.lr
EPOCHS = args.e
lambda_coord = args.lc
lambda_obj = args.lo
lambda_noobj = args.lno

# ==================== DISPOSITIVO PARA TREINO ============================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (f'device: {device}')
# ==================== DISPOSITIVO PARA TREINO ============================================

# ==================== PREPARAÇÃO DO DATASET ==============================================
train_dataset, test_dataset, train_dataloader, test_dataloader = preparar_dataset_e_dataloaders(
    S, C, B, IMG_SIZE, BATCH_SIZE, TEST_SIZE)
# ==================== PREPARAÇÃO DO DATASET ==============================================

# ==================== PREPARAÇÃO DA FUNÇÃO PERDA =========================================
yolo_loss = YOLO_LOSS(S, B, C, IMG_SIZE, lambda_coord, lambda_obj, lambda_noobj)
yolo_loss.to(device)
# ==================== PREPARAÇÃO DA FUNÇÃO PERDA =========================================

# =========================== CRIANDO O MODELO E OTIMIZADOR ===============================
ma = model_architeture()
model = YOLO(S, C, B, IMG_SIZE, ma.architecture_config)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# =========================== CRIANDO O MODELO E OTIMIZADOR ===============================

# ================================= TREINAMENTO  ==========================================
for epoch in range(EPOCHS):

    train_loss = []
    model.train()
    for imgs_tensor, target_tensor in tqdm(train_dataloader):
        imgs_tensor, target_tensor = imgs_tensor.to(device), target_tensor.to(device)
        
        model.zero_grad()
        predictions = model(imgs_tensor)
        loss = yolo_loss(predictions, target_tensor)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    
    train_loss = np.array(train_loss).mean()
    test_loss = validacao(model, yolo_loss, test_dataloader, device)
    print (f'Epoch: [{epoch}], train_loss: {round(train_loss, 3)}, test_loss: {round(test_loss, 3)}')
    salvar_checkpoint(model, epoch)
    _, _ = salvar_resultado_uma_epoca(model, test_dataset, epoch, device)

print ('fim treinamento')