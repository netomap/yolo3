from random import random, choice, randint
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw

class yolo_dataset(Dataset):

    def __init__(self, S, B, C, IMG_SIZE, imgs_list):
        self.imgs_list = imgs_list
        self.S = S
        self.B = B
        self.C = C
        self.img_size = IMG_SIZE
        self.annotations = pd.read_csv('annotations.csv')

        self.transformer = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
        ])

        self.inv_transformer = transforms.Compose([
            transforms.Normalize(mean=(-1., -1., -1.), std=(2., 2., 2.)),
            transforms.ToPILImage()
        ])
    
    def get_random_img_pil(self):
        return Image.open(choice(self.imgs_list))

    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        img_tensor = self.transformer(Image.open(img_path))
        annotations = self.annotations[self.annotations['img_path'] == img_path].values

        target_labels = self.preparar_anotacoes(annotations)
        
        return img_tensor, target_labels
    
    def preparar_anotacoes(self, annotations_):
        r"""
        Esta função prepara as anotações no formato target igual ao formato da predição da rede yolo.  
        O formato é [S, S, (1 + 4 + C)]
        """
        target_labels = torch.zeros((self.S, self.S, 5+self.C)) # [S, S, 5] onde 5 -> [p, xc, yc, w, h]

        for img_path, imgw, imgh, tipo, xc, yc, w, h in annotations_:
            
            j, i = int(self.S * xc), int(self.S * yc) # índices j, i da célula a qual percence o centro xc, yc, respectivamente
            xc_rel, yc_rel = self.S*xc - j, self.S*yc - i # posição relativa dos centros em comparação à posição x1, y1 da célula
            w_rel, h_rel = w*self.S, h*self.S # tamanhos w e h relativos à célula. 

            vetor_classes = np.zeros(self.C) # vetor original todo zerado que representa as posições das classes.
            vetor_classes[tipo] = 1 # coloca 1 no índice que representa aquela classe

            if (target_labels[i, j, 0] == 0):
                target_labels[i, j] = torch.hstack([torch.tensor([1]), torch.tensor(vetor_classes), torch.tensor([xc_rel, yc_rel, w_rel, h_rel])])
                #target_labels[i, j] = torch.tensor([1, xc_rel, yc_rel, w_rel, h_rel])
        
        return target_labels

def preparar_dataset_e_dataloaders(S, C, B, IMG_SIZE, BATCH_SIZE, test_size=0.1, print_debug=True):
    r"""
    Função que prepara os dados em treino-teste.

    Args: 
        S (grid), C (n de classes), B (bounding boxes), IMG_SIZE (redimensionamendo das imagens).  
        BATCH_SIZE (Tamanho do lote), test_size=0.1
    
    Returns: 
        Retorna uma tupla de quatro elementos: 
        train_dataset, test_dataset, train_dataloader, test_dataloader
    """
    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()
    imgs_train, imgs_test = train_test_split(imgs_list, test_size=test_size, shuffle=True)

    train_dataset = yolo_dataset(S, B, C, IMG_SIZE, imgs_train)
    test_dataset = yolo_dataset(S, B, C, IMG_SIZE, imgs_test)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if (print_debug):
        print (f'len_train_dataset: {len(train_dataset)}, len_test_dataset: {len(test_dataset)}')
        print (f'len_train_dataloader: {len(train_dataloader)}, len_test_dataloader: {len(test_dataloader)}')
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader

if __name__ == '__main__':

    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()
    dataset = yolo_dataset(4, 1, 2, 300, imgs_list)

    img_tensor, target_labels = choice(dataset)
    print (f'img_tensor.shape: {img_tensor.shape}, bbox_tensor.shape: {target_labels.shape}')