import numpy as np
import sklearn.neighbors
from carregar_dados import oecd_bli, gdp_per_capita
from combinar_dados import prepare_country_stats

#Preparando os dados para modelo de vizinhos mais proximos (KNN)


country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Selecionando o modelo
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

#treinando modelo
model.fit(x,y)

#fazendo a predição para a ilha do Chipre
x_new = [[22587]]
print(model.predict(x_new))
