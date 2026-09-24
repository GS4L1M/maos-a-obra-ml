#Grafico mostrando overfitting (sobreajuste): um modelo complicado demais que decora os dados
#em vez de aprender a tendência geral
#status_de_todos_paises = tabela com os 36 países
from carregar_dados import gdp_per_capita
from combinar_dados import preparar_gdp
from manipulando_dados import status_de_todos_paises
import numpy as np #contas com vetores
import matplotlib.pyplot as plt #gráficos
from sklearn import linear_model #modelos lineares
from sklearn import preprocessing #ferramentas pra transformar os dados antes do modelo
from sklearn import pipeline #junta várias etapas num modelo só

#pontos dos 36 países
status_de_todos_paises.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
plt.axis([0, 110000, 0, 10])

#dados de treino: PIB e satisfação dos 36 países
Xfull = np.c_[status_de_todos_paises["GDP per capita"]]
yfull = np.c_[status_de_todos_paises["Life satisfaction"]]

#PolynomialFeatures cria PIB², PIB³... até PIB^30, deixando a "reta" virar uma curva cheia de curvas
poly = preprocessing.PolynomialFeatures(degree=30, include_bias=False)
#StandardScaler coloca os números numa escala parecida (PIB^30 é um número gigante)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression() #regressão linear normal

#pipeline: os dados passam pelo poly, depois pelo scaler e depois vão pro modelo, tudo de uma vez
pipeline_reg = pipeline.Pipeline([('poly', poly), ('scal', scaler), ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull) #treina tudo

X = np.linspace(0, 110000, 1000) #1000 valores de PIB entre 0 e 110 mil
#X[:, np.newaxis] transforma a lista de 1000 valores numa tabela de 1 coluna, como o modelo espera
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve) #a curva passa perto dos pontos mas faz umas voltas que não fazem sentido
plt.xlabel("GDP per capita (USD)")
plt.show()

#outro exemplo de overfitting do livro: países com "W" no nome em inglês parecem ser muito felizes
#(New Zealand, Norway, Sweden, Switzerland), mas isso é coincidência e não uma regra de verdade
print(status_de_todos_paises.loc[[c for c in status_de_todos_paises.index if "W" in c.upper()]]["Life satisfaction"])

#e a regra falha fácil: vários países com W no nome não são nada ricos nem felizes
gdp_preparado = preparar_gdp(gdp_per_capita) #país vira o índice e a coluna 2015 vira GDP per capita
print(gdp_preparado.loc[[c for c in gdp_preparado.index if "W" in c.upper()]].head())
