#Exemplo 1-1 do livro: o programa completo de regressão linear (carregar, preparar, ver, treinar e prever)
#importando os dados crus da OCDE e do FMI do nosso modulo carregar_dados
from carregar_dados import oecd_bli, gdp_per_capita
#prepare_country_stats junta as duas tabelas e tira os países das pontas / preparar_gdp arruma a tabela do FMI
from combinar_dados import prepare_country_stats, preparar_gdp
import pandas as pd #tabelas
import numpy as np #contas com vetores
import matplotlib.pyplot as plt #gráficos

gdp_preparado = preparar_gdp(gdp_per_capita) #país vira o índice e a coluna 2015 vira GDP per capita
cyprus_gdp_per_capita = gdp_preparado.loc["Cyprus"]["GDP per capita"] #PIB do Chipre
print(cyprus_gdp_per_capita)

from sklearn import linear_model #modelos lineares do scikit-learn
#os dados já vêm carregados do carregar_dados (import lá em cima), não precisa ler os CSVs de novo

#preparando os dados: 29 países, x = PIB e y = satisfação (np.c_ deixa cada um como tabela de 1 coluna)
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#vendo os dados antes de treinar
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show() #o script para aqui até fechar a janela do gráfico

#escolhendo o modelo: regressão linear
model = linear_model.LinearRegression()

#treinando o modelo = achando a reta que passa mais perto dos pontos
model.fit(X, y)

#previsão pro Chipre, usando o PIB dele
X_new = [[22587]]  #PIB per capita do Chipre
print(model.predict(X_new)) #mostra [[5.96242338]]
