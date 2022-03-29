import torch
from torch import nn
import pandas as pd

class YOLO_LOSS(nn.Module):

    def __init__(self, S, B, C, IMG_SIZE, lambda_coord, lambda_obj, lambda_noobj):
        super(YOLO_LOSS, self).__init__()
        r"""
        Função que calcula a perda entre o predito pela rede YOLO e a sua target. 

        Args:  
            S: Número de Grids.
            B: Número de box por células.
            C: Número de classes do problema.
            IMG_SIZE: Valor que as imagens são redimensionadas.
            lambda_coord: Taxa de penalização para a perda das coordenadas.
            lambda_obj: Taxa de penalização para a perda quando encontra um objeto.
            lambda_noobj: Taxa de penalização para a perda quand não encontra objeto.
        
        Returns:  
            loss: o valor da função perda.
        
        """
        self.S = S
        self.B = B
        self.C = C
        self.img_size = IMG_SIZE
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        r"""
        Args: 
            predictions: Tensor que vem no formato flatten. [N, S*S*(5+C)]
            targets: vem no formato [N, S, S, (5+C)]
        
        Returns: 
            loss: o somatório de todas as perdas, ponderadas.
        """
        # aqui faz o reshape do predictions igual ao shape do target.
        predictions = predictions.reshape(targets.shape)

        # A variável exists_box é a que possui em qual célula está presente o objeto.
        exists_box = targets[:, :, :, 0].unsqueeze(3) # e dá uma dimensão a mais para facilitar o código a partir daqui
        no_exists_box = 1 - exists_box  # no_exists_box são as células que não possuem os centros dos objetos. 

        # ======================= CALCULO PERDA PARA COORDENADAS =============================
        # Aqui fazemos a multiplicação e pegamos as predições
        # somente das células que de fato são responsáveis pelo objeto
        box_predictions = exists_box * predictions[:, :, :, -4:] # pois xc, yc, w, h estão nas 4 últimas posições
        box_targets = exists_box * targets[:, :, :, -4:] # pois xc, yc, w, h estão nas 4 últimas posições

        # De acordo com a função perda do paper, o somatório de w e h
        # é feito pelas suas raizes quadradas. Assim vamos alterar somente
        # esses itens.

        # aqui fazemos:                torch.sign para respeitar o sinal    # raiz quadrada do absoluto + 1e-6 para não dar erro
        box_predictions[:,:,:,2:] = torch.sign(box_predictions[:,:,:,2:]) * torch.sqrt(torch.abs(box_predictions[:,:,:,2:] + 1e-6))
        box_targets[:,:,:,2:] = torch.sign(box_targets[:,:,:,2:]) * torch.sqrt(torch.abs(box_targets[:,:,:,2:] + 1e-6))
        # para esse segundo não é necessário, mas fez apenas para manter o padrão

        box_loss = self.mse(box_predictions.reshape(-1, 4), box_targets.reshape(-1, 4))
        # ======================= CALCULO PERDA PARA COORDENADAS =============================

        # ======================= CALCULO PERDA PARA CLASSES  ================================
        # No paper, a função perda para as classes é definida como um mse simples. Ao contrário
        # do que se usa para classificação de imagem (multiclasse), que normalmente é softmax.
        classes_predictions = predictions[:,:,:, 1:1+self.C]
        classes_targets = targets[:,:,:,1:1+self.C]
        classes_loss = self.mse(
            classes_predictions.reshape((-1, self.C)), 
            classes_targets.reshape((-1, self.C))
        )
        # ======================= CALCULO PERDA PARA CLASSES  ================================

        # ====================== CALCULO PARA PROBABILIDADE DE DETECAÇÃO DE OBJETO ===========
        prob_exists_prediction = exists_box * predictions[:,:,:,0].unsqueeze(3)
        prob_exists_target = exists_box * targets[:,:,:,0].unsqueeze(3)
        prob_exists_loss = self.mse(
            prob_exists_prediction.reshape(-1, 1), 
            prob_exists_target.reshape(-1, 1)
        )
        # ====================== CALCULO PARA PROBABILIDADE DE DETECAÇÃO DE OBJETO ===========

        # ================= CALCULO PARA PROBABILIDADE DE NÃO DETECAÇÃO DE OBJETO ============
        prob_no_exists_prediction = no_exists_box * predictions[:,:,:,0].unsqueeze(3)
        prob_no_exists_target = no_exists_box * targets[:,:,:,0].unsqueeze(3)
        prob_noobj_loss = self.mse(
            prob_no_exists_prediction.reshape(-1, 1),
            prob_no_exists_target.reshape(-1, 1)
        )
        # ================= CALCULO PARA PROBABILIDADE DE NÃO DETECAÇÃO DE OBJETO ============
        
        loss = (
            box_loss * self.lambda_coord
            + prob_exists_loss * self.lambda_obj
            + prob_noobj_loss * self.lambda_noobj
            + classes_loss
        )

        # print (f'box_loss: {box_loss.item()}, exists_loss: {prob_exists_loss.item()}, noobj_loss: {prob_noobj_loss.item()}, classes_loss: {classes_loss.item()}')
        # print (f'box_loss: {box_loss.item()*self.lambda_coord}, exists_loss: {prob_exists_loss.item()*self.lambda_obj}, noobj_loss: {self.lambda_noobj*prob_noobj_loss.item()}')

        return loss


if __name__ == '__main__':

    from model import YOLO
    from dataset import yolo_dataset
    from torch.utils.data import DataLoader
    from model_architecture import model_architeture

    S = 4
    C = 2
    B = 1
    IMG_SIZE = 300

    ma = model_architeture()
    model = YOLO(S, C, B, IMG_SIZE, ma.architecture_config)
    yolo_loss = YOLO_LOSS(S, B, C, IMG_SIZE, 1, 1, 1)
    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()
    dataset = yolo_dataset(S, B, C, IMG_SIZE, imgs_list)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    imgs_tensor, target_tensor = next(iter(dataloader))
    predictions = model(imgs_tensor)

    loss = yolo_loss(predictions, target_tensor)
    print (f'loss: {loss}')