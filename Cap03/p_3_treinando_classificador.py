#importando os dados do MNIST que foram carregados no p2
from p_2_minst import X, y, some_digit
from sklearn.linear_model import SGDClassifier
import numpy as np


#o MNIST já vem separado: as 60 mil primeiras imagens são de treino e as 10 mil últimas de teste
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
#classificador binário: o rótulo vira True quando é 5 e False para qualquer outro dígito
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


#SGD = gradiente descendente estocástico; o random_state fixa o sorteio para o resultado se repetir
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

#só roda quando executamos o p3 diretamente, e não quando os outros arquivos importam o sgd_clf
if __name__ == "__main__":
    print(sgd_clf)
    #pergunta ao modelo se o some_digit (aquele 5 do p2) é um 5
    print(f"o some_digit é um 5? {sgd_clf.predict([some_digit])}")
