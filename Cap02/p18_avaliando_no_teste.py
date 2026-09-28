from p10_conjunto_teste_estratificado import strat_test_set
from p15_atributos_texto_e_transformadores import full_pipeline
from p17_ajuste_fino import grid_search
from sklearn.metrics import mean_squared_error
from scipy import stats
import numpy as np

#avaliando o sistema no conjunto de teste
#o teste ficou guardado desde o p10 e ninguém mexeu nele, agora é a hora de usar, uma vez só, no final
#se ficarmos ajustando o modelo olhando o teste, ele deixa de ser um teste de verdade

#o modelo final é o melhor da busca em grade do p17
final_model = grid_search.best_estimator_

#separa as pistas (X) do gabarito (y), igual fizemos com o treino no p14
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

#ATENÇÃO: só transform, sem fit! o pipeline usa as medianas e medias aprendidas no treino
#se fizesse fit no teste, ele aprenderia com os dados do teste, e isso é trapaça
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

if __name__ == "__main__":
    #uns 47.900 de erro no teste
    print(final_rmse)

    #intervalo de confiança de 95%
    #o 47.900 é uma estimativa só, com outras casas de teste o numero seria um pouco diferente
    #o intervalo diz entre quais valores o erro verdadeiro deve estar, com 95% de confiança
    confidence = 0.95
    #o erro ao quadrado de cada casa do teste
    squared_errors = (final_predictions - y_test) ** 2
    #stats.t.interval calcula o intervalo usando a distribuição t de Student
    #len - 1 = graus de liberdade, loc = a media, scale = o erro padrão da media (stats.sem)
    #o np.sqrt no final volta de erro ao quadrado para dolares
    #resultado: entre uns 45.900 e 49.800
    print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                   loc=squared_errors.mean(),
                                   scale=stats.sem(squared_errors))))

    #o mesmo calculo feito na mão
    m = len(squared_errors)
    mean = squared_errors.mean()
    #ppf devolve o valor de t que deixa 97.5% da distribuição para trás (sobram 2.5% em cada ponta)
    tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
    #margem = t * desvio padrão / raiz do numero de casas
    tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
    print(np.sqrt(mean - tmargin), np.sqrt(mean + tmargin))

    #usando z-scores (distribuição normal) em vez de t-scores
    #com muitas casas (4128) a t e a normal ficam quase iguais, então o resultado é praticamente o mesmo
    zscore = stats.norm.ppf((1 + confidence) / 2)
    zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
    print(np.sqrt(mean - zmargin), np.sqrt(mean + zmargin))
