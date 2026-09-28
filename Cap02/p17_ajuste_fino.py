from p15_atributos_texto_e_transformadores import housing_labels, full_pipeline, housing_prepared, num_attribs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import pandas as pd

#mostra todas as colunas no print, sem esconder as do meio com ...
pd.set_option("display.max_columns", None)

#ajuste fino do modelo (fine-tune)
#o random forest do p16 foi o melhor, agora vamos procurar os melhores hiperparametros para ele
#hiperparametro = configuração que a gente escolhe antes de treinar (ex: quantas arvores), o modelo não aprende sozinho

#busca em grade (grid search)
#em vez de testar os valores na mão, o GridSearchCV testa todas as combinações com validação cruzada
param_grid = [
    #testa 12 (3×4) combinações de hiperparametros
    #n_estimators = quantas arvores, max_features = quantas colunas cada arvore pode olhar em cada divisão
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    #depois testa 6 (2×3) combinações com bootstrap=False (cada arvore usa todos os dados, sem sorteio)
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
#treina em 5 folds, totalizando (12+6)*5 = 90 rodadas de treinamento
#return_train_score=True guarda tambem a nota no treino, para comparar com a da validação
#n_jobs=-1 usa todos os nucleos do processador (não está no livro, é só para ir mais rapido)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, n_jobs=-1)
#fica fora do __main__ porque o p18 (conjunto de teste) e os exercicios usam o melhor modelo daqui
#no final ele treina de novo o melhor modelo com o treino inteiro (refit), pronto para usar
grid_search.fit(housing_prepared, housing_labels)

#analisando os melhores modelos e seus erros
#feature_importances_ = o quanto cada coluna ajudou nas previsões (a soma de todas dá 1)
feature_importances = grid_search.best_estimator_.feature_importances_
#o housing_prepared não tem nome de coluna, então montamos a lista dos nomes na mesma ordem das colunas
#as 8 numericas + as 3 combinadas do CombineAttributesAdder + as 5 categorias do one-hot
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#named_transformers_["cat"] pega o OneHotEncoder que está dentro do full_pipeline (o "cat" é o nome que demos no p15)
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

if __name__ == "__main__":
    #a melhor combinação encontrada: max_features=8 e n_estimators=30
    #30 é o maior valor que testamos, então talvez valha testar valores maiores
    print(grid_search.best_params_)
    #o modelo pronto com esses hiperparametros
    print(grid_search.best_estimator_)

    #a nota de cada combinação testada (RMSE, então quanto menor melhor)
    #zip anda nas duas listas ao mesmo tempo, pegando a nota e os parametros de cada combinação
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    #o melhor ficou em uns 49.900, melhor que os 50.400 do random forest padrão do p16

    #todos os resultados da busca numa tabela (tempo, notas de cada fold, parametros...)
    print(pd.DataFrame(grid_search.cv_results_))

    #busca aleatoria (randomized search)
    #quando são muitas combinações, em vez de testar todas ele sorteia algumas
    #randint(low=1, high=200) = sorteia um numero inteiro de 1 a 199
    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
    forest_reg = RandomForestRegressor(random_state=42)
    #n_iter=10 = testa 10 combinações sorteadas (cada uma em 5 folds)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                    random_state=42, n_jobs=-1)
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    #a melhor sorteada ficou em uns 49.100, melhor que a da grade, porque testou valores maiores

    #a importancia de cada coluna, sem nome fica dificil de entender
    print(feature_importances)
    #zip junta cada importancia com o nome da coluna, e o sorted com reverse=True põe a mais importante primeiro
    #o median_income é de longe a mais importante, e das categorias só a INLAND ajuda de verdade
    #então daria para tentar tirar as colunas que quase não ajudam (isso é o exercicio 3, no p22)
    for importancia, nome in sorted(zip(feature_importances, attributes), reverse=True):
        print(importancia, nome)
