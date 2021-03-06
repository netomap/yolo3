# YOLO (versão 1)

Este repositório é mais um que foi desenvolvido com objetivo de aprendizado para implementação de um modelo de detecção de objetos, utilizando a rede YOLO - versão 1. 

# Descrição

Foram feitas outras tentativas para aprendizado e essa é uma abordagem para implementar a rede de detecção de objetos.  
Foram feitas simplificações neste script, com intuito de aprendizado, como considerar 1 predição por cada "célula" das imagens, ou seja, B=1. 
Assim, a saída da rede neural tem o seguinte formato:  

![saida](./docs/formato_saida_classe_dataset_py.png)
Onde S representa o número de grids que a imagem vai *"ser dividida"*.  
C0, C1... Cn os índices das classes a qual o objeto pertence. Colocar no índice o valor 1 para a qual a classe pertence.  
Xc, Yc representam os valores relativos do centro do objeto à posição X1, Y1 da célula *"responsável"* pelo objeto. Variam de 0 a 1.  
W, H representam o tamanho do objeto e possuem valores relativos à imagem. Podem variar de 0 e serem maiores que 1, uma vez que o objeto pode ser maior que a "célula" responsável por ele.  

# Resultados

- Alguns resultados ficaram muito bons, embora precisem ser treinados por mais tempo. 

![output1](./imgs_results/output1.png)  
![output2](./imgs_results/output2.png)  
![output3](./imgs_results/output3.png)  
![output4](./imgs_results/output4.png)  

- Já para algumas imagens onde os objetos estão muito próximos, não foi possível detectá-los, conforme exibido abaixo:  

![output5](./imgs_results/output5.png)  
![output6](./imgs_results/output6.png)  



# Fonte
- [Paper Original](https://arxiv.org/pdf/1506.02640.pdf)  
- Uma paylist no YOUTUBE explicando de forma excelente os conceitos de deteção de objetos além da implementação da rede YOLO utilizando Pytorch: [Introduction to Object Detection in Deep Learning
](https://www.youtube.com/watch?v=t-phGBfPEZ4&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq)
