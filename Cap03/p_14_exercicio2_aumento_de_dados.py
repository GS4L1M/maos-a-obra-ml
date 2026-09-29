#EXERCÍCIO 2: aumento de dados (data augmentation)
#criar cópias do treino movidas 1 pixel para cada lado e treinar o melhor modelo do exercício 1 com elas
from p_1_configuracao_inicial import save_fig
from p_3_treinando_classificador import X_train, X_test, y_train, y_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#no livro é scipy.ndimage.interpolation, mas esse caminho foi removido das versões novas do scipy
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np


#recebe uma imagem (linha de 784 pixels), move dx para os lados e dy para cima/baixo
#o reshape([-1]) volta a imagem para uma linha só
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


if __name__ == "__main__":
    #testando a função numa imagem: original, 5 pixels para baixo e 5 para a esquerda
    image = X_train[1000]
    shifted_image_down = shift_image(image, 0, 5)
    shifted_image_left = shift_image(image, -5, 0)

    plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.title("Original", fontsize=14)
    plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
    plt.subplot(132)
    plt.title("Shifted down", fontsize=14)
    plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
    plt.subplot(133)
    plt.title("Shifted left", fontsize=14)
    plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
    save_fig("shifted_images_plot")
    plt.show()

    #começa com as imagens originais e acrescenta 4 cópias de cada (direita, esquerda, baixo, cima)
    X_train_augmented = [image for image in X_train]
    y_train_augmented = [label for label in y_train]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)

    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)

    #embaralha para as cópias não ficarem todas juntas no final
    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_idx]
    y_train_augmented = y_train_augmented[shuffle_idx]

    #no livro é KNeighborsClassifier(**grid_search.best_params_), mas importar o grid_search
    #do p13 faria a busca rodar de novo; então copiei aqui o resultado que ela encontrou
    best_params = {'n_neighbors': 4, 'weights': 'distance'}
    knn_clf = KNeighborsClassifier(**best_params, n_jobs=-1)
    knn_clf.fit(X_train_augmented, y_train_augmented)

    #o livro avisa que essa parte pode levar cerca de uma hora
    y_pred = knn_clf.predict(X_test)
    #só aumentando os dados, a acurácia sobe uns 0,5%
    print(f"acurácia no teste com aumento de dados: {accuracy_score(y_test, y_pred)}")
