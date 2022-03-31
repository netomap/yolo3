import torch
from torch import nn
from torchvision.ops.boxes import box_iou, _box_cxcywh_to_xyxy
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
from torch.functional import F

def salvar_checkpoint(model, epochs):

    if (not(os.path.exists('./models/'))):
        os.mkdir('./models/')
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'datetime': datetime.now(),
        'S': model.S, 'B': model.B, 'C': model.C, 'IMG_SIZE': model.IMG_SIZE, 'epochs': epochs,
        'descricao': str(model),
        'architecture_config': model.architecture_config
    }
    torch.save(checkpoint, f'./models/checkpoint_{epochs}_epochs.pth')

def predict(model, img_pil, transformer, prob_threshold, print_grid):
    r"""
    Função que retorna a img_pil com as anotações preditas desenhadas e suas bboxes.

    Args:  
        model: modelo yolo.
        img_pil: image PIL simples.  
        transformer: o transformador de imagepil para tensor, que é utilizado no dataset.
        prob_threshold: limite mínimo de probabilidade de objeto para plotar uma anotação. 
        print_grid: Default False. Desenhar as grids da visão yolo.
    
    Returns:  
        imgs_pil: Image pil com as anotações desenhadas. 
        bboxes: o bboxes das anotações. [prob_obj, prob_class, ind_class, x1, y1, x2, y2] em formato normal e absoluto.

    """
    model.eval()
    img_tensor = transformer(img_pil)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)
    
    predictions = predictions[0].detach().cpu()
    predictions = predictions.reshape((model.S, model.S, model.C+5))
    img_pil, bboxes = desenhar_anotacoes(img_pil, predictions, model.S, model.C, prob_threshold=prob_threshold, print_grid=print_grid)

    return img_pil, bboxes

def salvar_resultado_uma_epoca(model, dataset, epoch, device, prob_threshold, save_img=True):
    r"""
    Faz uma simples predição em uma imagem do conjunto de teste.  
    E salva na pasta ./results a imagem com as bboxes preditas pelo modelo.
    
    Args:  
        model: modelo yolo.
        dataset: Preferência pelo dataset do conjunto teste.
        epoch: Número da época em questão.
        device: cuda:0 ou cpu
        save_img: default True. Salva a imagem no diretório ./results/
    
    Returns: 
        bboxes: As anotações da predição do modelo.

    """
    if (not(os.path.exists('./results/'))):
        os.mkdir('./results')

    model.eval()
    img_pil = dataset.get_random_img_pil()
    img_tensor = dataset.transformer(img_pil)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)
    
    predictions = predictions[0].detach().cpu()
    S = model.S
    C = model.C
    predictions = predictions.reshape((S, S, (5+C)))
    img_pil, bboxes = desenhar_anotacoes(img_pil, predictions, S, C, prob_threshold=prob_threshold)

    if (save_img):
        img_pil.save(f'./results/result_{epoch}_epoch.jpg')

    return img_pil, bboxes

def desenhar_anotacoes(img_pil, predictions_tensor, S, C, prob_threshold = 0.5, print_grid=False):
    r"""
    Função que pega uma img_pil simples e desenha as anotações a partir de predictions_tensor.

    Args:  
        img_pil: Image PIL normal.  
        predictions_tensor: Tensor de dimensão [S, S, (5 + C)] que pode vir da rede neural ou do próprio yolo_dataset (target)
        S: Número de grids.  
        C: Número de classes. 
        prob_threshold: Limiar para considerar se tem ou não objeto. É predito pelo modelo e vem na primeira posição do bbox. 
        pring_grid: Default false. Serve para imprimir as grids que a imagem é dividida 'virtualmente'
    
    Returns: 
        img_pil: Mesma img_pil com suas anotações.
        bboxes: Vetor numpy com as anotações transformadas em [prob_obj, prob_class, ind_class, x1, y1, x2, y2] em formato normal e absoluto.
    """
    assert predictions_tensor.shape == torch.rand((S, S, (5+C))).shape, "Tensor com shape inválido"

    predictions_tensor = predictions_tensor.detach().cpu().numpy()

    imgw, imgh = img_pil.size
    cell_size = 1 / S # tamanho de cada célula em percentual
    draw = ImageDraw.Draw(img_pil)

    if (print_grid):
        for k in range(S):
            draw.line([k*cell_size*imgw, 0, k*cell_size*imgw, imgh], fill='red', width=1)
            draw.line([0, k*cell_size*imgh, imgw, k*cell_size*imgh], fill='red', width=1)
    
    contador = 0
    bboxes = []
    for l in range(S):
        for c in range(S):
            prob_obj = predictions_tensor[l, c, 0] # probabilidade de existir um objeto
            classes = predictions_tensor[l, c, 1:1+C] # pega os índices das classes
            classes = F.softmax(torch.tensor(classes), dim=-1) # transforma os valores para softmax
            prob_class, indice_class = classes.max(0) # retorna dois tensores. A probabilidade e o índice que tem esse maior valor
            prob_class = prob_class.item()    # apenas retornando os valores normais, não o tensor
            indice_class = indice_class.item() # o mesmo de cima

            if (prob_class >= prob_threshold):
                contador += 1
                xc_rel, yc_rel, w_rel, h_rel = predictions_tensor[l,c,1+C:]*cell_size # pega os quatro últimos valores que são coordenadas
                x1_cell, y1_cell = c*cell_size, l*cell_size                           # e também multiplica por cell_size
                xc, yc = x1_cell + xc_rel, y1_cell + yc_rel         ##
                w, h = w_rel, h_rel             #### Valores ainda em percentuais
                x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2     ##

                x1, y1, x2, y2 = x1*imgw, y1*imgh, x2*imgw, y2*imgh
                bboxes.append([prob_obj, prob_class, indice_class, x1, y1, x2, y2])

                draw.rectangle([x1, y1, x2, y2], fill=None, width=1, outline='black')
                draw.rectangle([x1, y1-15, x2, y1], fill='black')
                draw.text([x1+2, y1-13], f'{indice_class}:{round(100*prob_class)}%', fill='white')

    return img_pil, bboxes

def validacao(model, loss_fn, dataloader, device):
    r"""
    Retorna o valor da perda entre o predito pela rede e o anotado pelo dataset. É uma média das perdas de cada lote.
    Args:  
        model: modelo YOLO.
        loss_fn: função perda, instanciada no treinamento.
        dataloader: é o dataloader do teste.
        device: cuda:0 ou cpu
    
    Returns:  
        Tensor numérico, valor médio das perdas dos lotes do dataloader do conjunto de dados teste. 

    """
    print ('validação: ')
    model.eval()
    test_loss = []
    with torch.no_grad():
        for imgs_tensor, target_tensor in tqdm(dataloader):
            imgs_tensor, target_tensor = imgs_tensor.to(device), target_tensor.to(device)
            output = model(imgs_tensor)
            loss = loss_fn(output, target_tensor)
            test_loss.append(loss.item())
    
    return np.array(test_loss).mean()


def calculate_ious(predictions, targets):
    r"""
    Função que calcula os ious entre as predições e as targets (anotações).  
    Os tensores vêm no formato [S, S, 5+C] onde 5+C-> [p, C1, C2...Cn, xc, yc, w, h]

    Args:  
        Predictions -> Tensor que vem no formato [S, S, 5+C]
        Target -> Tensor que vem no formato [S, S, 5+C]
    
    Returns: 
        ious -> Tensor no formato [S, S, 1] onde 1 representa o IOU entre as bboxes entre cada casas.
    """
    assert predictions.shape[:2] == targets.shape[:2], "Shapes diferentes"  # verificando os shapes se possuem a mesma dimensão de grid
    assert predictions.shape[-1] == targets.shape[-1], "Shapes diferentes"    

    bbox1 = predictions[..., -4:] # pega somente [xc, yc, w, h] pois vem no formato [p, C1, C2...Cn, xc, yc, w, h]
    bbox2 = targets[..., -4:] # o mesmo

    # agora convertendo para [x1, y1, x2, y2]
    bbox1 = _box_cxcywh_to_xyxy(bbox1)
    bbox2 = _box_cxcywh_to_xyxy(bbox2)

    # calculando agora os ious
    ious = box_iou(bbox1.reshape(-1, 4), bbox2.reshape(-1, 4)) # essa função nativa do pytorch só aceita tensores com shape= [N, 4]
    ious = ious.diagonal() # assim pegamos sua diagonal pois queremos somente os elementos comparados na mesma casa
    
    ious = ious.reshape(predictions.shape[:2]) # retorna com o mesmo shape de linhas e colunas S, S e mais uma dimensão com o valor do iou

    return ious

def inspecao_visual_ious(bbox1, bbox2):
    
    bbox1 = bbox1[..., 1:] # pega somente [xc, yc, w, h] pois vem no formato [p, xc, yc, w, h]
    bbox2 = bbox2[..., 1:] # o mesmo

    # agora convertendo para [x1, y1, x2, y2]
    bbox1 = _box_cxcywh_to_xyxy(bbox1)
    bbox2 = _box_cxcywh_to_xyxy(bbox2)

    maximo = int(torch.tensor([bbox1.max(), bbox2.max()]).max())
    img_pil = Image.new('RGB', size=(maximo+20, maximo+20), color='white')
    draw = ImageDraw.Draw(img_pil)

    for k, (box1, box2) in enumerate(zip(bbox1.reshape(-1, 4), bbox2.reshape(-1, 4))):
        box1 = box1.detach().numpy()
        box2 = box2.detach().numpy()

        draw.rectangle(box1, fill=None, outline='red')
        draw.text([box1[0],box1[1]-13], str(k), fill='red')

        draw.rectangle(box2, fill=None, outline='green')
        draw.text([box2[0],box2[1]-13], str(k), fill='green')
    
    return img_pil