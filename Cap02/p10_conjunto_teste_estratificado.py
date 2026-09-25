from p02_baixar_dados import load_housing_data
from p01_configuracao_inicial import save_fig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
housing = load_housing_data()

#jeito 4 (o que o livro usa no resto do capitulo): amostragem estratificada
#a renda (median_income) é o que mais ajuda a prever o preço das casas
#então o teste precisa ter a mesma proporção de renda baixa, media e alta que a base inteira
#pra isso divide as casas em faixas de renda (estratos) e sorteia 20% de cada faixa

#Conferindo a mediana do dataframe
#quase todo mundo fica entre 1.5 e 6 (renda em dezenas de milhares de dolares, 3.5 = 35 mil)
#e depois tem uma cauda comprida até 15, isso ajuda a escolher as faixas
housing["median_income"].hist()
save_fig("Mediana")
plt.show()

#pd.cut transforma a renda (numero quebrado) em 5 faixas
#bins são os limites: 0 a 1.5 = faixa 1, 1.5 a 3 = faixa 2 ... 6 pra cima (np.inf = infinito) = faixa 5
#o primeiro limite tem que ser 0, se for 0.5 as 12 casas com renda menor ficam sem faixa (NaN) e o sklearn trava
#6 limites formam 5 faixas, uma pra cada label
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
print(housing["income_cat"].value_counts())#quantas casas em cada faixa (822, 6581, 7236, 3639, 2362)
housing["income_cat"].hist()
save_fig("income_cat")
plt.show()

#StratifiedShuffleSplit sorteia respeitando a proporção de cada faixa
#n_splits=1 = quero só uma divisão, test_size=0.2 = 20% pro teste, random_state=42 = sorteio sempre igual
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#o split.split recebe os dados e a coluna das faixas e devolve os indices (numeros das linhas) de treino e teste
#o for roda uma vez só por causa do n_splits=1, e o .loc pega as linhas com esses indices
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#conferindo: a proporção de cada faixa no teste tem que ser quase igual à da base inteira
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(housing["income_cat"].value_counts() / len(housing))

#comparando o estratificado com o sorteio comum (train_test_split)
#devolve a porcentagem de casas em cada faixa (quantas tem na faixa / total)
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#monta uma tabela com uma coluna pra cada jeito, o sort_index deixa as faixas em ordem 1 a 5
#o sort_index vai depois de fechar o DataFrame, porque é metodo da tabela e não do dicionario
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
#erro em % comparado com a base inteira, ex: 100 * 35.85 / 35.06 = 102.2, tirando 100 = 2.2% de erro
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
#resultado: o sorteio comum erra até 5% em algumas faixas e o estratificado menos de 0.4%
print(compare_props)
