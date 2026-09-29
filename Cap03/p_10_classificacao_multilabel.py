from p_2_minst import some_digit
from p_3_treinando_classificador import X_train, y_train
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
import numpy as np


#multilabel: cada imagem recebe mais de um rótulo ao mesmo tempo
y_train_large = (y_train >= 7)      #rótulo 1: o dígito é grande (7, 8 ou 9)?
y_train_odd = (y_train % 2 == 1)    #rótulo 2: o dígito é ímpar?
#np.c_ junta as duas colunas lado a lado: cada linha vira [grande?, ímpar?]
y_multilabel = np.c_[y_train_large, y_train_odd]

#o KNN olha as imagens mais parecidas do treino e aceita vários rótulos de uma vez
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


if __name__ == "__main__":
    #o some_digit é um 5: não é grande (False) e é ímpar (True)
    print(f"previsão para o some_digit [grande?, ímpar?]: {knn_clf.predict([some_digit])}")

    #f1 de cada rótulo, depois a média dos dois (average="macro")
    #o n_jobs=-1 usa todos os núcleos do processador; o livro avisa que pode levar muito tempo
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
    print(f"f1 médio dos dois rótulos: {f1_score(y_multilabel, y_train_knn_pred, average='macro')}")
