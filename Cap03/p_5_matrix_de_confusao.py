
from p_3_treinando_classificador import sgd_clf, X_train, y_train_5
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#igual ao cross_val_score, mas devolve a previsão de cada imagem em vez da acurácia
#cada imagem é prevista por um modelo que não a viu no treino (cv=3 divide em 3 partes)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#só roda quando executamos o p5 diretamente, o p6 importa só o y_train_pred
if __name__ == "__main__":
    #linhas = gabarito (não é 5 / é 5), colunas = previsão do modelo (não é 5 / é 5)
    #[[verdadeiros negativos, falsos positivos],
    # [falsos negativos,      verdadeiros positivos]]
    matriz = confusion_matrix(y_train_5, y_train_pred)
    print(f"o resultado da matriz de confusão é:\n{matriz}")

    #um modelo perfeito teria previsões iguais ao gabarito
    y_train_perfect_predictions = y_train_5  # fingindo que chegamos à perfeição
    #na matriz dele, só a diagonal principal tem valores e os dois tipos de erro ficam zerados
    print(confusion_matrix(y_train_5, y_train_perfect_predictions))
