from p10_conjunto_teste_estratificado import strat_train_set
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

#mostra todas as colunas no print, sem esconder as do meio com ...
pd.set_option("display.max_columns", None)

#mesmo housing do p14: o treino sem a coluna do preço
housing = strat_train_set.drop("median_house_value", axis=1)

#lidando com atributos de texto e categoricos
#o ocean_proximity é texto (INLAND, NEAR BAY...) e os algoritmos só fazem conta com numero
#os colchetes duplos [[ ]] pegam a coluna como tabela (DataFrame), que é o formato que o sklearn espera
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)#no notebook isso aparecia sozinho, no .py não mostra nada, por isso o print embaixo
print(housing_cat.head(10))

#OrdinalEncoder troca cada categoria por um numero (0, 1, 2, 3, 4)
#fit_transform = fit (aprende quais categorias existem) + transform (troca o texto pelo numero) de uma vez
Ordinal_enconder = OrdinalEncoder()
housing_cat_encoded= Ordinal_enconder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
#a lista das categorias que ele aprendeu, na ordem: a posição na lista é o numero que ela virou
print(Ordinal_enconder.categories_)
#problema: o algoritmo acha que 0 e 1 são "parecidos" e 0 e 4 "distantes", o que não é verdade aqui
#(<1H OCEAN = 0 e NEAR OCEAN = 4 são bem parecidos na vida real)

#solução: one-hot, cria uma coluna para cada categoria, com 1 na categoria da casa e 0 nas outras
#assim nenhuma categoria fica "mais perto" da outra
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot#mesma coisa do head(10) lá em cima, no .py não mostra nada
#o resultado é uma matriz esparsa (guarda só onde tem 1, pra economizar memoria)
#toarray() transforma numa matriz normal do numpy só para conseguirmos ver
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)


#transformadores customizados
#o sklearn deixa a gente criar nosso proprio transformador, que funciona igual o SimpleImputer e o OneHotEncoder
#aqui ele faz as combinações de atributos do p13 (comodos por casa, pessoas por casa, quartos por comodo)
#INDICE DAS COLUNAS: a posição de cada coluna na matriz do numpy (que não tem nome de coluna, só numero)
rooms_ix, bedrooms_ix, population_ix, househlds_ix = 3,4,5,6
#BaseEstimator dá de graça o get_params e set_params (usados depois para ajustar hiperparametros)
#TransformerMixin dá de graça o fit_transform, só precisamos escrever o fit e o transform
class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    #add_bedrooms_per_room é um hiperparametro: liga ou desliga a coluna de quartos por comodo
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    #fit não precisa aprender nada aqui, mas tem que existir e devolver self para funcionar no pipeline
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        #x[:, n] = todas as linhas (:) da coluna n
        rooms_per_household = x[:, rooms_ix] / x[:, househlds_ix]
        population_per_household = x[:, population_ix] / x[:, househlds_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
            #np.c_ gruda as colunas novas do lado direito da matriz x
            return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[x, rooms_per_household, population_per_household]

attr_adder = CombineAttributesAdder(add_bedrooms_per_room=False)
#.values transforma a tabela do pandas numa matriz do numpy (por isso usamos os indices numericos)
housing_extra_attribs = attr_adder.transform(housing.values)
print(housing_extra_attribs[:5])
#Observe que fixei os índices (3, 4, 5, 6) diretamente no código para concisão e clareza no livro, mas seria muito mais limpo obtê-los dinamicamente, desta forma:
#get_loc procura a posição da coluna pelo nome, assim se a ordem das colunas mudar o codigo não quebra
col_names = "total_rooms", "total_bedrooms","population", "households"
rooms_ix, bedrooms_ix, population_ix, househlds_ix = [housing.columns.get_loc(c) for c in col_names]

#renomeando as colunas
#a matriz do numpy volta a ser tabela do pandas, com os nomes antigos + os 2 nomes novos
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
print(housing_extra_attribs.head())


#pipelines de transformação
#pipeline = uma fila de etapas, a saída de uma vira a entrada da próxima, como uma linha de montagem
#em vez de chamar imputer, attr_adder e scaler um por um, chamamos o pipeline uma vez só
#e ele aplica exatamente as mesmas etapas depois no conjunto de teste e nos dados novos

#o pipeline numerico só recebe colunas de numero, então tiramos o ocean_proximity (igual no p14)
housing_num = housing.drop("ocean_proximity", axis=1)

#cada etapa é uma dupla ("nome que a gente escolhe", transformador)
#todas as etapas menos a ultima precisam ter fit_transform, a ultima pode ser qualquer estimador
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),#1) troca os vazios pela mediana (p14)
        ('attribs_adder', CombineAttributesAdder()),#2) cria as colunas combinadas (a classe de cima)
        ('std_scaler', StandardScaler()),#3) padroniza: cada coluna fica com media 0 e desvio padrão 1
    ])
#escalonamento: sem ele, colunas com numeros grandes (população na casa dos milhares)
#pesam mais que colunas com numeros pequenos (renda de 0 a 15) em muitos algoritmos

#fit_transform no pipeline chama o fit_transform de cada etapa em sequencia
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)

#ColumnTransformer: um pipeline só para as colunas de numero e outro só para as de texto, tudo junto
#list(housing_num) devolve a lista com os nomes das colunas numericas
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

#cada dupla vira trio: ("nome", transformador, lista de colunas em que ele vai trabalhar)
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),#colunas de numero passam pelo pipeline numerico
        ("cat", OneHotEncoder(), cat_attribs),#a coluna de texto passa pelo one-hot
    ])

#ele aplica cada transformador nas suas colunas e gruda os resultados lado a lado
#recebe o housing completo (com o texto) e devolve tudo pronto para o modelo
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
#16512 casas e 16 colunas: 8 numericas + 3 combinadas + 5 do one-hot
print(housing_prepared.shape)
