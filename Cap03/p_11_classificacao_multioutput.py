from p_1_configuracao_inicial import save_fig
from p_2_minst import plot_digit
from p_3_treinando_classificador import X_train, X_test
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


#multioutput: cada imagem tem vários rótulos e cada rótulo pode ter vários valores
#aqui o modelo vai limpar o ruído: a saída são os 784 pixels, cada um de 0 a 255

#sujando as imagens com ruído aleatório (números de 0 a 99 somados em cada pixel)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
#o gabarito é a imagem limpa original
y_train_mod = X_train
y_test_mod = X_test


if __name__ == "__main__":
    #esquerda: imagem com ruído (entrada) / direita: imagem limpa (o que queremos obter)
    some_index = 0
    plt.subplot(121); plot_digit(X_test_mod[some_index])
    plt.subplot(122); plot_digit(y_test_mod[some_index])
    save_fig("noisy_digit_example_plot")
    plt.show()

    #o KNN aprende a sair da imagem suja e chegar na limpa
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digit(clean_digit)
    save_fig("cleaned_digit_example_plot")
    plt.show()
