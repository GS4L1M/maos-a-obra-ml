import numpy as np #contas com vetores
import sklearn.linear_model #modelos lineares do scikit-learn
from carregar_dados import oecd_bli, gdp_per_capita #dados crus da OCDE e do FMI
from combinar_dados import prepare_country_stats #junta as tabelas e tira os países das pontas

#Preparando os dados para modelo de regressão

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita) #29 países com PIB e satisfação
#np.c_ transforma a coluna em tabela de 1 coluna, que é o formato que o sklearn espera
x = np.c_[country_stats["GDP per capita"]] #entrada: PIB
y = np.c_[country_stats["Life satisfaction"]] #saída: satisfação

# Selecionando o modelo
model = sklearn.linear_model.LinearRegression() #traça uma reta passando pelos pontos

#Treinando o modelo
model.fit(x,y) #acha a reta que passa mais perto dos 29 países

#criando a predição para a ilha do Chipre
x_new = [[22587]] #Chipre (GDP per capita)
print(model.predict(x_new)) #previsão de satisfação com a vida no Chipre (uns 5.96)
