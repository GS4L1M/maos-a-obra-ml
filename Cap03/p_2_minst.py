#importando o dataset mnist_784 (70 mil imagens de dígitos escritos à mão, de 28x28 pixels)
from p_1_configuracao_inicial import save_fig
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#as_frame=False faz os dados virem como array do numpy em vez de DataFrame do pandas
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

#separando as imagens (X) e os rotulos (y) do dataset
#cada linha de X é uma imagem com 784 colunas (28x28 pixels, com valores de 0 a 255)
X, y = mnist["data"], mnist["target"]

some_digit = X[0] # o X[0] representa a primeira imagem do data set se fosse X [1] buscaria a segunda e assim por diante

#fazendo a correção dos rotulos, que vêm como texto ('5' entre aspas), para número inteiro
#sem isso, o y_train == 5 do p3 daria False para todas as imagens
y = y.astype(np.uint8)

#mostra uma imagem só (recebe uma linha de X)
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

# EXTRA
#mostra várias imagens juntas numa grade (10 por linha)
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # Isto equivale a n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Acrescenta imagens vazias para completar o final da grade, se necessário:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Muda o formato do array para ele virar uma grade de imagens 28×28:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combina os eixos 0 e 2 (eixo vertical da grade de imagens e eixo vertical da imagem),
    # e os eixos 1 e 3 (eixos horizontais). Primeiro precisamos colocar os eixos que
    # queremos combinar um ao lado do outro, usando transpose(), e só então
    # podemos usar o reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Agora que temos uma imagem grande, só precisamos mostrá-la:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

#tudo que está aqui dentro só roda quando executamos o p2 diretamente,
#quando outro arquivo faz "from p2_minst import ..." essa parte é pulada
if __name__ == "__main__":
    #Verificando o shape x e y do dataset
    print(X.shape)
    print(y.shape)

    some_digit_image = some_digit.reshape(28, 28) #o reshape reorganiza os 784 valores da linha numa grade de 28x28 para virar imagem
    plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    save_fig("some_digit_plot")
    plt.show()

    #VISUALIZANDO O ROTULO DE Y (depois do astype aparece 5 sem aspas)
    print(y[0])

    #as 100 primeiras imagens do dataset
    plt.figure(figsize=(9,9))
    example_images = X[:100]
    plot_digits(example_images, images_per_row=10)
    save_fig("more_digits_plot")
    plt.show()
