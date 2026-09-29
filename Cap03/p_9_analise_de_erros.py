from p_1_configuracao_inicial import save_fig
from p_2_minst import plot_digits
from p_3_treinando_classificador import X_train, y_train
from p_8_classificacao_multiclasse import sgd_clf_multi, X_train_scaled
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# desde o sklearn 0.22, dá para usar o sklearn.metrics.plot_confusion_matrix()
#versão colorida da matriz, com a barra de cores do lado (o livro não usa, fica de opção)
def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


if __name__ == "__main__":
    #mesma ideia do p5, só que agora com as 10 classes (demora uns minutos)
    y_train_pred = cross_val_predict(sgd_clf_multi, X_train_scaled, y_train, cv=3)
    #matriz 10x10: linha = dígito real, coluna = dígito previsto
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)

    #a mesma matriz como imagem: quanto mais claro, maior o número
    #a diagonal clara quer dizer que a maioria das imagens foi classificada certo
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    save_fig("confusion_matrix_plot", tight_layout=False)
    plt.show()

    #cada linha é dividida pelo total de imagens daquele dígito, para comparar taxas de erro
    #(sem isso, um dígito que aparece mais pareceria errar mais)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums

    #zera a diagonal (os acertos) para sobrar só os erros
    #a coluna do 8 fica clara: muitos dígitos são confundidos com 8
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    save_fig("confusion_matrix_errors_plot", tight_layout=False)
    plt.show()

    #olhando de perto a confusão entre 3 e 5
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)] #3 previstos como 3
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)] #3 previstos como 5
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)] #5 previstos como 3
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)] #5 previstos como 5

    #os quadros da diagonal (esquerda em cima e direita embaixo) são acertos, os outros dois são erros
    plt.figure(figsize=(8,8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    save_fig("error_analysis_digits_plot")
    plt.show()
