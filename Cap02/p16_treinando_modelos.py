from p15_atributos_texto_e_transformadores import housing, housing_labels, full_pipeline, housing_prepared
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

#selecionar e treinar um modelo
#os dados já estão prontos (housing_prepared do p15), agora é só escolher o modelo e treinar
#igual no capitulo 1: cria o modelo, fit para aprender e predict para prever

#mostra o resultado da validação cruzada: as notas de cada rodada, a media e o desvio padrão
#fica fora do __main__ porque o p17 tambem usa
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

if __name__ == "__main__":
    #treinando e avaliando no conjunto de treinamento
    #1) regressão linear, a mesma do capitulo 1 (lá era o PIB prevendo a satisfação, aqui 16 colunas prevendo o preço)
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    #testando o pipeline completo em 5 casas do treino
    #iloc[:5] pega as 5 primeiras linhas pela posição
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    #só transform (sem fit): o pipeline já aprendeu as medianas e as medias no p15, agora só aplica
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:", lin_reg.predict(some_data_prepared))
    #comparando as previsões com os valores reais
    print("Labels:", list(some_labels))
    print(some_data_prepared)

    #RMSE (raiz do erro quadratico medio): em media quanto o modelo erra, na mesma unidade do preço (dolares)
    #mean_squared_error faz a media dos erros ao quadrado e o np.sqrt tira a raiz
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    #deu uns 68 mil dolares de erro, sendo que a maioria das casas vale entre 120 e 265 mil: modelo fraco
    #isso é underfitting (subajuste), o modelo é simples demais para os dados
    print(lin_rmse)

    #MAE (erro absoluto medio): media dos erros sem elevar ao quadrado, pesa menos os erros muito grandes
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print(lin_mae)

    #2) arvore de decisão, um modelo mais poderoso que consegue achar relações não lineares
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    #deu 0.0: erro zero! desconfie, é quase certeza que o modelo decorou os dados (overfitting, sobreajuste)
    #o problema é que estamos avaliando nos mesmos dados que ele treinou, igual fazer a prova com o gabarito
    print(tree_rmse)

    #melhor avaliação usando validação cruzada (cross-validation)
    #o k-fold divide o treino em 10 partes (folds), treina em 9 e avalia na que sobrou, 10 vezes
    #cada rodada deixa uma parte diferente de fora, e no fim temos 10 notas em vez de uma só
    #o sklearn espera uma função de "utilidade" (quanto maior melhor), por isso usa o MSE negativo
    #e por isso o -scores antes da raiz, para voltar a ser positivo
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    #agora a arvore erra uns 71 mil em media: pior que a regressão linear! o 0.0 era mesmo overfitting
    display_scores(tree_rmse_scores)

    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    #3) random forest (floresta aleatoria): muitas arvores de decisão treinadas em pedaços sorteados dos dados
    #a previsão final é a media de todas as arvores, isso é chamado de ensemble learning
    #n_estimators=100 = 100 arvores
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    #uns 18 mil no treino
    print(forest_rmse)

    #n_jobs=-1 usa todos os nucleos do processador ao mesmo tempo (não está no livro, é só para ir mais rapido)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10, n_jobs=-1)
    forest_rmse_scores = np.sqrt(-forest_scores)
    #uns 50 mil na validação cruzada: o melhor até agora
    #mas 18 mil no treino contra 50 mil na validação mostra que ainda tem overfitting
    display_scores(forest_rmse_scores)

    #o describe do pandas mostra o resumo das 10 notas (media, desvio, minimo, maximo...) de uma vez
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    print(pd.Series(np.sqrt(-scores)).describe())

    #4) SVR (maquina de vetores de suporte para regressão) com kernel linear
    #uns 111 mil de erro, bem pior (os hiperparametros dele são vistos nos exercicios, no p20 e p21)
    svm_reg = SVR(kernel="linear")
    svm_reg.fit(housing_prepared, housing_labels)
    housing_predictions = svm_reg.predict(housing_prepared)
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    print(svm_rmse)
