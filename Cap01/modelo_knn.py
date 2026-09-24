import numpy as np #contas com vetores
import sklearn.neighbors #modelos de vizinhos mais próximos do scikit-learn
from carregar_dados import oecd_bli, gdp_per_capita #dados crus da OCDE e do FMI
from combinar_dados import prepare_country_stats #junta as tabelas e tira os países das pontas

#Preparando os dados para modelo de vizinhos mais proximos (KNN)


country_stats = prepare_country_stats(oecd_bli, gdp_per_capita) #29 países com PIB e satisfação
#np.c_ transforma a coluna em tabela de 1 coluna, que é o formato que o sklearn espera
x = np.c_[country_stats["GDP per capita"]] #entrada: PIB
y = np.c_[country_stats["Life satisfaction"]] #saída: satisfação

# Selecionando o modelo
#KNN não traça reta: procura os 3 países com PIB mais parecido e tira a média da satisfação deles
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

#treinando modelo (no KNN o treino é só guardar os dados pra consultar depois)
model.fit(x,y)

#fazendo a predição para a ilha do Chipre
x_new = [[22587]] #PIB do Chipre
print(model.predict(x_new)) #uns 5.77 = média de Eslovênia, Portugal e Espanha
