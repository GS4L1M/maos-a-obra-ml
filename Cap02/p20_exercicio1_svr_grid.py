from p15_atributos_texto_e_transformadores import housing_labels, housing_prepared
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np

#Exercicio 1
#Pergunta: experimente um regressor de Maquina de Vetores de Suporte (sklearn.svm.SVR) com varios hiperparametros,
#como kernel="linear" (com varios valores para o C) ou kernel="rbf" (com varios valores para o C e o gamma).
#Não se preocupe com o significado desses hiperparametros por enquanto. Qual é o desempenho do melhor SVR?

#AVISO: no livro esta busca leva uns 30 minutos. Com n_jobs=-1 (todos os nucleos do processador) fica bem mais rapido,
#mas ainda demora alguns minutos. O verbose=2 vai mostrando cada treino no terminal, para ver que não travou.

#a grade tem 8 combinações com kernel linear + 7×6 = 42 com kernel rbf = 50 combinações
#com 5 folds, são 250 treinos
#kernel = o "formato" que o SVR usa para separar os dados: linear = reta, rbf = curvas
#C = o quanto ele pode errar (C grande = tenta errar pouco no treino), gamma = o quanto as curvas do rbf podem ser apertadas
param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

if __name__ == "__main__":
    svm_reg = SVR()
    grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                               verbose=2, n_jobs=-1)
    grid_search.fit(housing_prepared, housing_labels)

    #o melhor modelo, avaliado com validação cruzada de 5 folds
    #best_score_ é o MSE negativo (igual no p16), então -negative_mse e a raiz para voltar a dolares
    negative_mse = grid_search.best_score_
    rmse = np.sqrt(-negative_mse)
    #uns 70.300: bem pior que o random forest (uns 49.900 no p17)
    print(rmse)

    #os melhores hiperparametros: kernel linear com C=30000
    #o C que ganhou é o MAIOR valor da grade, e quando isso acontece o melhor provavelmente está mais para frente
    #o certo seria rodar de novo tirando os C pequenos e colocando valores maiores
    print(grid_search.best_params_)
