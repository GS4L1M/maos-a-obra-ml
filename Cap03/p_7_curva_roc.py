from p_1_configuracao_inicial import save_fig
from p_3_treinando_classificador import X_train, y_train_5
from p_6_precisao_renovacao import y_scores, recall_90_precision
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


#a curva ROC mostra a taxa de verdadeiros positivos (revocação) contra a taxa de falsos positivos
#a linha tracejada é um classificador que chuta aleatório; quanto mais longe dela (para cima e à esquerda), melhor
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # diagonal tracejada
    plt.axis([0, 1, 0, 1])                                    # Não aparece no livro
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Não aparece
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Não aparece
    plt.grid(True)                                            # Não aparece


#tudo que está aqui dentro só roda quando executamos o p7 diretamente,
#o p12 importa só a função plot_roc_curve
if __name__ == "__main__":
    #fpr = taxa de falsos positivos, tpr = taxa de verdadeiros positivos, para cada limiar
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    plt.figure(figsize=(8, 6))                                    # Não aparece
    plot_roc_curve(fpr, tpr)
    #o ponto vermelho é o limiar de 90% de precisão que achamos no p6
    fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Não aparece
    plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Não aparece
    plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Não aparece
    plt.plot([fpr_90], [recall_90_precision], "ro")               # Não aparece
    save_fig("roc_curve_plot")                                    # Não aparece
    plt.show()

    #AUC = área embaixo da curva ROC: 1 é perfeito, 0.5 é chute
    print(f"AUC do SGD: {roc_auc_score(y_train_5, y_scores)}")

    #agora um RandomForest para comparar com o SGD
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    #o RandomForest não tem decision_function, então pedimos a probabilidade de cada classe
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")
    #coluna 0 = chance de não ser 5, coluna 1 = chance de ser 5; a coluna 1 vira a pontuação
    y_scores_forest = y_probas_forest[:, 1] # pontuação = probabilidade da classe positiva
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    #revocação do RandomForest com a mesma taxa de falsos positivos do ponto do SGD
    recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
    plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
    plt.plot([fpr_90], [recall_90_precision], "ro")
    plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
    plt.plot([fpr_90], [recall_for_forest], "ro")
    plt.grid(True)
    plt.legend(loc="lower right", fontsize=16)
    save_fig("roc_curve_comparison_plot")
    plt.show()

    #a curva do RandomForest fica bem mais perto do canto de cima, então a AUC é maior
    print(f"AUC do RandomForest: {roc_auc_score(y_train_5, y_scores_forest)}")

    #precisão e revocação do RandomForest usando o predict normal (limiar de 50% de chance)
    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print(f"precisão do RandomForest: {precision_score(y_train_5, y_train_pred_forest)}")
    print(f"revocação do RandomForest: {recall_score(y_train_5, y_train_pred_forest)}")
