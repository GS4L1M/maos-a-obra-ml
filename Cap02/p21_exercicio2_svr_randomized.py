from p15_atributos_texto_e_transformadores import housing_labels, housing_prepared
from p01_configuracao_inicial import save_fig
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
import matplotlib.pyplot as plt
import numpy as np

#Exercicio 2
#Pergunta: tente substituir o GridSearchCV por RandomizedSearchCV.

#AVISO: no livro esta busca leva uns 45 minutos. Com n_jobs=-1 fica bem mais rapido, mas ainda demora alguns minutos.

#consulte https://docs.scipy.org/doc/scipy/reference/stats.html
#para a documentação do expon() e reciprocal() e de outras distribuições de probabilidade
#em vez de uma lista de valores (como no p20), cada hiperparametro recebe uma distribuição para sortear
#Nota: o gamma é ignorado quando o kernel é 'linear'
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),#sorteia entre 20 e 200000, sem preferir nenhuma escala (explicado lá embaixo)
        'gamma': expon(scale=1.0),#sorteia valores perto de 1, mas às vezes bem maiores ou menores
    }

if __name__ == "__main__":
    svm_reg = SVR()
    #n_iter=50 = 50 combinações sorteadas, × 5 folds = 250 treinos (o mesmo tanto do p20)
    rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                    n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                    verbose=2, random_state=42, n_jobs=-1)
    rnd_search.fit(housing_prepared, housing_labels)

    #o melhor modelo, avaliado com validação cruzada de 5 folds
    negative_mse = rnd_search.best_score_
    rmse = np.sqrt(-negative_mse)
    #uns 54.700: agora sim ficou bem mais perto do random forest (mas ainda não chegou lá)
    print(rmse)

    #os melhores hiperparametros: desta vez achou um bom conjunto para o kernel rbf
    #com o mesmo numero de treinos, a busca aleatoria costuma achar hiperparametros melhores que a grade
    #esses valores são copiados no p23 (exercicio 4), para não precisar rodar esta busca de novo
    print(rnd_search.best_params_)

    #a distribuição exponencial que usamos para o gamma, com scale=1.0
    #rvs sorteia 10000 valores para ver o formato no histograma
    #alguns valores são bem maiores ou menores que 1, mas olhando o logaritmo (grafico da direita)
    #a maioria fica entre exp(-2) e exp(+2), ou seja, entre uns 0.1 e 7.4
    expon_distrib = expon(scale=1.)
    samples = expon_distrib.rvs(10000, random_state=42)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)#1 linha, 2 colunas, grafico 1 (o da esquerda)
    plt.title("Exponential distribution (scale=1.0)")
    plt.hist(samples, bins=50)
    plt.subplot(122)#grafico 2 (o da direita)
    plt.title("Log of this distribution")
    plt.hist(np.log(samples), bins=50)
    save_fig("distribuicao_exponencial")
    plt.show()

    #a distribuição que usamos para o C é bem diferente: o logaritmo (grafico da direita) fica quase plano,
    #ou seja, todas as escalas (dezenas, centenas, milhares...) têm a mesma chance de serem sorteadas
    #o reciprocal é bom quando não temos ideia da escala do hiperparametro,
    #e o expon é bom quando já sabemos mais ou menos a escala
    reciprocal_distrib = reciprocal(20, 200000)
    samples = reciprocal_distrib.rvs(10000, random_state=42)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.title("Reciprocal distribution (scale=1.0)")
    plt.hist(samples, bins=50)
    plt.subplot(122)
    plt.title("Log of this distribution")
    plt.hist(np.log(samples), bins=50)
    save_fig("distribuicao_reciprocal")
    plt.show()
