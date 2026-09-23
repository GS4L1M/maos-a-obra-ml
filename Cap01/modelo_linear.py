import numpy as np
import sklearn.linear_model
from carregar_dados import oecd_bli, gdp_per_capita
from combinar_dados import prepare_country_stats

#Preparando os dados para modelo de regressão

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Selecionando o modelo
model = sklearn.linear_model.LinearRegression()

#Treinando o modelo
model.fit(x,y)

#criando a predição para a ilha do Chipre
x_new = [[22587]] #Chipre (GDP per capita)
print(model.predict(x_new)) #previsão de satisfação com a vida no Chipre