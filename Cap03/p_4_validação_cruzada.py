from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone, BaseEstimator
from p_3_treinando_classificador import sgd_clf, X_train, y_train_5
import numpy as np


#classificador que sempre responde "não é 5", usado como base de comparação
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros(len(X), dtype=bool)


#tudo que está aqui dentro só roda quando executamos o p4 diretamente,
#quando outro arquivo faz "from p_4_validação_cruzada import ..." essa parte é pulada
if __name__ == "__main__":
    #validação cruzada feita na mão: divide o treino em 3 partes mantendo a proporção de 5 em cada uma
    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        #cópia sem treino do modelo, para cada rodada começar do zero
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_folds = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        #acurácia = acertos / total de imagens do fold
        n_correct = sum(y_pred == y_test_folds)
        print(f"o resultado da medida de desempenho é {n_correct / len(y_pred)}")

    never_5_clf = Never5Classifier()
    resultado_never_5 = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print(f"o valor do teste no never classifier é: {resultado_never_5}")
