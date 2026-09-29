from p_2_minst import some_digit
from p_3_treinando_classificador import sgd_clf, X_train, y_train
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np


#cópia sem treino do sgd_clf do p3; se usássemos o próprio sgd_clf, o fit abaixo
#trocaria o modelo "é 5 ou não" por um de 10 classes em todo arquivo que importasse ele
sgd_clf_multi = clone(sgd_clf)
#agora o treino usa o y_train com os 10 dígitos, e não o y_train_5
sgd_clf_multi.fit(X_train, y_train)

#o StandardScaler deixa cada pixel com média 0 e desvio padrão 1
#o astype(np.float64) é para a conta não ficar limitada a números inteiros
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


#tudo que está aqui dentro só roda quando executamos o p8 diretamente,
#o p9 importa o sgd_clf_multi e o X_train_scaled daqui
if __name__ == "__main__":
    #o SVC é binário, mas quando recebe 10 classes o sklearn treina um modelo para cada par
    #de dígitos (0 contra 1, 0 contra 2...): isso se chama OvO, um contra um (45 modelos)
    #usamos só as 1000 primeiras imagens porque o SVC é lento
    svm_clf = SVC(gamma="auto", random_state=42)
    svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train, e não y_train_5
    print(f"previsão do SVC para o some_digit: {svm_clf.predict([some_digit])}")

    #uma pontuação por classe; a maior é a classe escolhida
    some_digit_scores = svm_clf.decision_function([some_digit])
    print(f"pontuações das 10 classes:\n{some_digit_scores}")
    print(f"posição da maior pontuação: {np.argmax(some_digit_scores)}")
    #o classes_ guarda as classes em ordem; aqui a posição 5 coincide com o dígito 5
    print(f"classes do modelo: {svm_clf.classes_}")
    print(f"classe na posição 5: {svm_clf.classes_[5]}")

    #forçando o OvR (um contra o resto): um modelo por dígito, "é 3 ou não é 3", total de 10
    ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
    ovr_clf.fit(X_train[:1000], y_train[:1000])
    print(f"previsão do OvR: {ovr_clf.predict([some_digit])}")
    print(f"quantidade de modelos dentro do OvR: {len(ovr_clf.estimators_)}")

    #o SGD já sabe lidar com várias classes (por dentro ele usa OvR)
    print(f"previsão do SGD multiclasse: {sgd_clf_multi.predict([some_digit])}")
    print(f"pontuações do SGD:\n{sgd_clf_multi.decision_function([some_digit])}")

    #acurácia nas 10 classes, antes e depois de escalar os dados (as duas linhas demoram bastante)
    print("acurácia sem escalar:", cross_val_score(sgd_clf_multi, X_train, y_train, cv=3, scoring="accuracy"))
    print("acurácia com os dados escalados:", cross_val_score(sgd_clf_multi, X_train_scaled, y_train, cv=3, scoring="accuracy"))
