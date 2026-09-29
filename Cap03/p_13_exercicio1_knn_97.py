#EXERCÍCIO 1: um classificador do MNIST com mais de 97% de acurácia no teste
#dica do livro: KNeighborsClassifier ajustando os hiperparâmetros weights e n_neighbors
from p_3_treinando_classificador import X_train, X_test, y_train, y_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    #3 valores de vizinhos x 2 tipos de peso = 6 combinações, cada uma testada em 5 folds (30 treinos)
    param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

    knn_clf = KNeighborsClassifier()
    #o livro avisa que pode levar umas 16 horas; o n_jobs=-1 usa todos os núcleos e deixa bem mais rápido
    #o verbose=3 vai mostrando cada treino no terminal, para dar para acompanhar
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"melhores hiperparâmetros: {grid_search.best_params_}")
    print(f"melhor acurácia na validação cruzada: {grid_search.best_score_}")

    #o grid_search já vem treinado com a melhor combinação, então dá para prever direto
    y_pred = grid_search.predict(X_test)
    print(f"acurácia no teste: {accuracy_score(y_test, y_pred)}")
