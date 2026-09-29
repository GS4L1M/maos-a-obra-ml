from p_1_configuracao_inicial import save_fig
from p_2_minst import some_digit, plot_digit
from p_3_treinando_classificador import X_train, X_test, y_train, y_test, y_train_5
from p_7_curva_roc import plot_roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, accuracy_score
#no livro é scipy.ndimage.interpolation, mas esse caminho foi removido das versões novas do scipy
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np


#move a imagem dx pixels para os lados e dy para cima/baixo; o espaço que sobra é preenchido com "new"
def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)


if __name__ == "__main__":
    #CLASSIFICADOR DUMMY (aleatório)
    #ele ignora as imagens e só chuta com base na proporção de 5 no treino
    dmy_clf = DummyClassifier(strategy="prior")
    y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_dmy = y_probas_dmy[:, 1]

    #a curva ROC dele fica em cima da diagonal: é o pior resultado possível
    fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
    plot_roc_curve(fprr, tprr)
    save_fig("roc_curve_dummy_plot")
    plt.show()

    #CLASSIFICADOR KNN
    #weights='distance': os vizinhos mais perto pesam mais no voto
    knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4, n_jobs=-1)
    knn_clf.fit(X_train, y_train)
    y_knn_pred = knn_clf.predict(X_test)
    print(f"acurácia do KNN no teste: {accuracy_score(y_test, y_knn_pred)}")

    #exemplo do shift: o 5 andou 5 pixels para a direita e 1 para baixo, com fundo cinza (100)
    plot_digit(shift_digit(some_digit, 5, 1, new=100))
    plt.show()

    #aumento de dados: cada imagem ganha 4 cópias, uma movida 1 pixel para cada lado
    X_train_expanded = [X_train]
    y_train_expanded = [y_train]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
        X_train_expanded.append(shifted_images)
        y_train_expanded.append(y_train)

    #junta tudo numa matriz só: 60 mil viram 300 mil imagens
    X_train_expanded = np.concatenate(X_train_expanded)
    y_train_expanded = np.concatenate(y_train_expanded)
    print(f"formato do treino aumentado: {X_train_expanded.shape}, {y_train_expanded.shape}")

    #treinando de novo com o treino aumentado (a acurácia deve subir um pouco)
    knn_clf.fit(X_train_expanded, y_train_expanded)
    y_knn_expanded_pred = knn_clf.predict(X_test)
    print(f"acurácia do KNN com aumento de dados: {accuracy_score(y_test, y_knn_expanded_pred)}")

    #um dígito difícil do teste: a chance que o modelo dá para cada classe (0 a 9)
    ambiguous_digit = X_test[2589]
    print(f"probabilidades do dígito ambíguo:\n{knn_clf.predict_proba([ambiguous_digit])}")
    plot_digit(ambiguous_digit)
    plt.show()
