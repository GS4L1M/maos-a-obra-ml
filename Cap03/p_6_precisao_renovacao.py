from p_1_configuracao_inicial import save_fig
from p_3_treinando_classificador import sgd_clf, X_train, y_train_5
from p_5_matrix_de_confusao import y_train_pred
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Não aparece no livro
    plt.xlabel("Threshold", fontsize=16)        # Não aparece
    plt.grid(True)                              # Não aparece
    plt.axis([-50000, 50000, 0, 1])             # Não aparece

#outro jeito de ver a troca: precisão direto contra a revocação, sem o limiar no meio
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


#em vez da previsão (True/False), pede a pontuação que o modelo dá para cada imagem
#acima do limiar (threshold) ele diz que é 5, abaixo diz que não é
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
#calcula a precisão e a revocação para todos os limiares possíveis
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#np.argmax acha o primeiro limiar em que a precisão chega a 90%
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


#tudo que está aqui dentro só roda quando executamos o p6 diretamente,
#o p7 importa o y_scores e o recall_90_precision daqui sem repetir os prints e os gráficos
if __name__ == "__main__":
    #precisão: de tudo que o modelo disse que é 5, quanto realmente era 5
    precisao = precision_score(y_train_5, y_train_pred)
    print(f"precisão: {precisao}")

    #a mesma conta feita na mão pela matriz de confusão
    #cm[1, 1] = verdadeiros positivos, cm[0, 1] = falsos positivos, cm[1, 0] = falsos negativos
    cm = confusion_matrix(y_train_5, y_train_pred)
    print(f"precisão na mão: {cm[1, 1] / (cm[0, 1] + cm[1, 1])}")

    #revocação: de todos os 5 que existem, quantos o modelo conseguiu achar
    revocacao = recall_score(y_train_5, y_train_pred)
    print(f"revocação: {revocacao}")
    print(f"revocação na mão: {cm[1, 1] / (cm[1, 0] + cm[1, 1])}")

    #f1 junta precisão e revocação num número só (média harmônica)
    #ele só fica alto se as duas forem altas
    f1 = f1_score(y_train_5, y_train_pred)
    print(f"f1: {f1}")
    print(f"f1 na mão: {cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)}")

    plt.figure(figsize=(8, 4))                                                                  # Não aparece
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Não aparece
    plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Não aparece
    plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Não aparece
    plt.plot([threshold_90_precision], [0.9], "ro")                                             # Não aparece
    plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Não aparece
    save_fig("precision_recall_vs_threshold_plot")                                              # Não aparece
    plt.show()

    #confere se usar o limiar 0 dá as mesmas previsões do p_5 (o SGD usa 0 por padrão)
    print(f"limiar 0 igual ao predict: {(y_train_pred == (y_scores > 0)).all()}")

    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls)
    #linhas vermelhas marcando o ponto em que a precisão chega a 90%
    plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
    plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
    plt.plot([recall_90_precision], [0.9], "ro")
    save_fig("precision_vs_recall_plot")
    plt.show()

    #usando o limiar de 90% de precisão (o threshold_90_precision já foi calculado lá em cima)
    #em vez do predict, a gente mesmo decide: pontuação acima do limiar = é 5
    y_train_pred_90 = (y_scores >= threshold_90_precision)

    #a precisão sobe para uns 90%, mas a revocação cai bastante
    print(f"precisão com limiar de 90%: {precision_score(y_train_5, y_train_pred_90)}")
    print(f"revocação com limiar de 90%: {recall_score(y_train_5, y_train_pred_90)}")
