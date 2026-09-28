# Importar as principais bibliotecas
# A versão do python precisa ser superior a 3.5

import sys
assert sys.version_info >=(3, 5) # garante que a versão python seja a 3.5
#o assert para o programa com erro se a condição for falsa, é um jeito de conferir a versão antes de começar

#importar e verificar a versão do scikitlearn
import sklearn
assert sklearn.__version__ >= "0.20"

# importando a biblioteca numpy
import numpy as np
import os

# para plotar os gráficos
import matplotlib as mpl
import matplotlib.pyplot as plt
#tamanho das letras dos nomes dos eixos e dos numeros do eixo x e y, vale pra todos os graficos do capitulo
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize =12)
mpl.rc('ytick', labelsize='12')

#salvar as figuras do capitulo
#as imagens vão pra pasta images/end_to_the_project, o "." quer dizer a pasta de onde o script é rodado (a raiz)
PROJECT_ROOT_DIR = "."
CHAPTER_ID ="end_to_the_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)#cria a pasta, o exist_ok=True evita erro se ela já existir

#salva o grafico que está aberto como png
#fig_id é o nome do arquivo, tight_layout ajusta as margens pra nada ficar cortado e resolution=300 deixa a imagem nitida
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Salvando imagem", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
