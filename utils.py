import torch
from torch import nn
from torchvision.ops.boxes import box_iou, _box_cxcywh_to_xyxy, nms
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

def predict(model, img_pil, transformer, class_threshold, iou_threshold, print_grid):
    r"""
    Função que retorna a img_pil com as anotações preditas desenhadas e suas bboxes.

    Args:  
        model: modelo yolo.
        img_pil: image PIL simples.  
        transformer: o transformador de imagepil para tensor, que é utilizado no dataset.
        class_threshold: limite mínimo de probabilidade de objeto para plotar uma anotação. 
        iou_threshold: limite mínimo de IOU para eliminar os bbox com menor confiança.
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
    
    predictions = predictions[0].detach().cpu().numpy()
    predictions = predictions.reshape((model.S, model.S, model.C+5))
    bboxes = converter_predicoes_em_bboxes(predictions, model.S, model.C, class_threshold, iou_threshold)
    img_pil = desenhar_anotacoes(img_pil, bboxes, model.S, print_grid)

    return img_pil, bboxes

def salvar_resultado_uma_epoca(model, dataset, epoch, class_threshold, iou_threshold, save_img=True):
    r"""
    Faz uma simples predição em uma imagem do conjunto de teste.  
    E salva na pasta ./results a imagem com as bboxes preditas pelo modelo.
    
    Args:  
        model: modelo yolo.
        dataset: Preferência pelo dataset do conjunto teste.
        epoch: Número da época em questão.
        class_threshold: limite mínimo para considerar a classe detectada.
        iou_threshold: limite mímino para excluir o bbox em regiões próximas.
        save_img: default True. Salva a imagem no diretório ./results/
    
    Returns: 
        None

    """
    if (not(os.path.exists('./results/'))):
        os.mkdir('./results')

    img_pil = dataset.get_random_img_pil()
    img_pil, _ = predict(model, img_pil, dataset.transformer, class_threshold, iou_threshold, False)

    if (save_img):
        img_pil.save(f'./results/result_{epoch}_epoch.jpg')

def converter_predicoes_em_bboxes(predictions, S, C, class_threshold, iou_threshold):
    r"""
    Esta função converte as predições em bboxes.  
    Também vai calcular o nms para retirar bboxes do mesmo objeto. 

    Args:  
        predictions: tensor (cpu) no formato que sai da rede YOLO. [S, S, (4 + 1 + C)]
        S: Número de Grids
        B: Número de bounding boxes
        C: Número de classes
        obj_threshold: limite mínimo para considerar que tem um objeto detectado.
        class_threshold: limite mínimo para considerar a classe detectada.
        iou_threshold: limite mínimo para considerar a interceção e eliminar o bbox de menor confiança.
    
    Returns:  
        bboxes: list que tem dimensão [n, 6] onde n => número de detecções e  
        6 => [confianca, ind_classe, x1, y1, x2, y2] em formato percentual.

    """
    cell_size = 1 / S # tamanho de cada célula em percentual

    bboxes = []
    for l in range(S):
        for c in range(S):
            prob_obj = predictions[l, c, 0] # probabilidade de existir um objeto
            classes = predictions[l, c, 1:1+C] # pega os índices das classes
            classes = F.softmax(torch.tensor(classes), dim=-1) # transforma os valores para softmax
            prob_class, indice_class = classes.max(0) # retorna dois tensores. A probabilidade e o índice que tem esse maior valor
            prob_class = prob_class.item()    # apenas retornando os valores normais, não o tensor
            indice_class = indice_class.item() # o mesmo de cima

            if (prob_class >= class_threshold):
                xc_rel, yc_rel, w_rel, h_rel = predictions[l,c,1+C:]*cell_size # pega os quatro últimos valores que são coordenadas
                x1_cell, y1_cell = c*cell_size, l*cell_size                    # e também multiplica por cell_size
                xc, yc = x1_cell + xc_rel, y1_cell + yc_rel         ##
                w, h = w_rel, h_rel             #### Valores ainda em percentuais
                x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2     ##

                bboxes.append([prob_obj, prob_class, indice_class, x1, y1, x2, y2])
    
    bboxes = np.array(bboxes)  # conversão para numpy
    indices = list(set(bboxes[:,2].astype(np.int))) # pegando todos os índices distintos de classes da lista
    bboxes_por_classes = []
    for ind in indices:
        bboxes_aux = bboxes[bboxes[:,2] == ind] # pega o índice da classe
        scores_aux = bboxes_aux[:,1]            # pega a confiança da predição
        bboxes_por_classes.append([ind, bboxes_aux[:,-4:], scores_aux])  # prepara uma lista com [indice, bboxes, confiança]

    bbox_final = []
    for ind_classe, bboxes_predict, scores_predict in bboxes_por_classes:
        indices_restantes = nms(torch.tensor(bboxes_predict), torch.tensor(scores_predict), iou_threshold=iou_threshold).detach().cpu().numpy()
        # calcula o nms a partir dos bboxes para cada tipo de classe. 
        scores_aux = scores_predict[indices_restantes]  # o resultado é indices_restantes, que usamos para pegar os scores
        bboxes_aux = bboxes_predict[indices_restantes]  # e os bboxes que são dos índices restantes.
        for score, bbox in zip(scores_aux, bboxes_aux):
            bbox_final.append(np.hstack([[score, ind_classe], bbox])) # adiciona no formato de uma dimensão somente
    
    return np.array(bbox_final)

def desenhar_anotacoes(img_pil, anotacoes, S, print_grid=False):
    r"""
    Desenha todas as anotações que são postas no vetor anotações. 

    Args:  
        img_pil: a imagem para desenhar as anotações.  
        anotações: vetor de dimensão [n, 6] => [confiança, indice_classe, x1, y1, x2, y2] no formato percentual.
    
    Returns: 
        img_pil: Image PIL com as anotações desenhadas.

    """
    draw = ImageDraw.Draw(img_pil)
    imgw, imgh = img_pil.size
    cell_size = 1 / S # percentual

    if (print_grid):
        for k in range(S):
            draw.line([k*cell_size*imgw, 0, k*cell_size*imgw, imgh], fill='black', width=1)
            draw.line([0, k*cell_size*imgh, imgw, k*cell_size*imgh], fill='black', width=1)
    
    for p_class, ind_class, x1, y1, x2, y2 in anotacoes:
        x1, y1, x2, y2 = x1*imgw, y1*imgh, x2*imgw, y2*imgh
        draw.rectangle([x1, y1, x2, y2], fill=None, outline='red')
        draw.rectangle([x1, y1-15, x1+40, y1], fill='red')
        draw.text([x1+2, y1-13], f'{int(ind_class)}-{round(100*p_class)}%', fill='white')

    return img_pil

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