from p10_conjunto_teste_estratificado import strat_train_set
from sklearn.impute import SimpleImputer
import pandas as pd

#mostra todas as colunas no print, sem esconder as do meio com ...
pd.set_option("display.max_columns", None)

#preparar os dados para os algoritmos de machine learning
#separa o que o modelo vai usar (as pistas) do que ele vai prever (a resposta), igual pergunta e gabarito
#drop tira a coluna do preço (axis=1 = coluna) e devolve uma tabela nova, o strat_train_set não muda
#drop e copy são metodos da tabela, por isso não precisam de import
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()#os rotulos (labels): só o preço

#limpeza de dados
#No livro ensina 3 modos de limpar os dados
#o total_bedrooms tem 158 casas vazias (NaN) no treino e a maioria dos algoritmos não funciona com valor vazio
#para demonstrar essas funções iremos criar uma copia do dataset
#isnull().any(axis=1) marca as linhas que têm pelo menos um valor vazio, e o head pega só as 5 primeiras
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)

#opção 1 remover linhas com valores nulos
#resultado: tabela vazia, as 5 casas somem inteiras só porque falta um dado
print(sample_incomplete_rows.dropna(subset=["total_bedrooms"]))

#opção 2 remover o atributo por completo
#resultado: as casas ficam mas a coluna total_bedrooms some pra todo mundo, até pra quem tinha o dado
print(sample_incomplete_rows.drop("total_bedrooms", axis=1))

#opção 3 usar a mediana para subistituir os valores auzentes
#resultado: as casas e a coluna ficam, e o NaN vira 433 (a mediana), é a que perde menos informação
#as opções 1 e 2 só mostram o resultado, essa muda a tabela de verdade por causa do =
#no pandas 3 o fillna(median, inplace=True) numa coluna não preenche nada, por isso atribui de volta com =
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"] = sample_incomplete_rows["total_bedrooms"].fillna(median)
print(sample_incomplete_rows)

#o livro faz a opção 3 com o SimpleImputer do sklearn
#a vantagem é que ele guarda a mediana do treino e usa a mesma depois no teste e nos dados novos

#removendo atributo de texto com sklearn  pois a mediana só pode ser calculada em atributos numéricos
housing_num = housing.drop("ocean_proximity", axis=1)

#cria o imputer a partir da "forma" SimpleImputer, dizendo que é pra preencher com a mediana
imputer = SimpleImputer(strategy="median")
#fit = aprender, ele calcula a mediana de cada coluna (mesmo das que não têm vazio, pra estar pronto pros dados novos)
imputer.fit(housing_num)
print(imputer.statistics_)#as medianas que ele guardou, uma por coluna
#verificando se o resultado é o mesmo do cálculo manual da mediana
print(housing_num.median().values)

#transformando o conjunto de treinamento
#transform = aplicar, troca cada vazio pela mediana da coluna
#o resultado X é uma matriz do numpy, sem nome de coluna e sem indice
X = imputer.transform(housing_num)
#pd.DataFrame monta a tabela do pandas de volta, colocando o nome das colunas e o indice (numero das linhas) originais
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
#conferindo as 5 casas que tinham vazio, agora com 433 no total_bedrooms
print(housing_tr.loc[sample_incomplete_rows.index.values])
print(imputer.strategy)#mostra a estrategia usada (median)
#o livro monta a mesma tabela de novo usando o indice do housing_num, dá o mesmo resultado porque o indice é igual
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
print(housing_tr.head())
