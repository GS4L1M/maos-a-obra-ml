#Grafico utilizando para demonstrar o apredinzado baseado em modelos usando regressão linear
#importando os dados crus da OCDE e do FMI do nosso modulo carregar_dados
from carregar_dados import oecd_bli, gdp_per_capita
#função que junta as duas tabelas e tira os países das pontas
from combinar_dados import prepare_country_stats
#tabelas já prontas do manipulando_dados (sample_data = os 29 países que vão pro gráfico)
#obs: importar esse arquivo roda os print de lá, por isso aparecem tabelas no terminal
from manipulando_dados import remove_indices, keep_indices, sample_data, status_de_todos_paises
import pandas as pd #tabelas
import numpy as np #contas com vetores
import matplotlib.pyplot as plt #gráficos

from sklearn import linear_model #modelos lineares do scikit-learn
lin1 = linear_model.LinearRegression() #cria o modelo de regressão linear (ainda sem treinar)
#np.c_ transforma a coluna em tabela de 1 coluna, que é o formato que o sklearn espera
Xsample = np.c_[sample_data["GDP per capita"]] #entrada: PIB
ysample = np.c_[sample_data["Life satisfaction"]] #saída: satisfação
lin1.fit(Xsample, ysample) #treina o modelo = acha a reta que passa mais perto dos pontos
#intercept_ = θ0 (onde a reta começa) e coef_ = θ1 (inclinação), os [0] tiram o número de dentro da lista
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
t0, t1 #no jupyter mostra os valores, aqui no .py não mostra nada (usar print pra ver)

#pontos dos 29 países, figsize=(5,3) é o tamanho da figura em polegadas
sample_data.plot(kind='scatter', x="GDP per capita", y="Life satisfaction", figsize=(5,3))
plt.xlabel("GDP per capita (USD)") #nome do eixo x
plt.axis([0, 60000, 0, 10]) #x de 0 a 60 mil e y de 0 a 10
X=np.linspace(0, 60000, 1000) #1000 valores de PIB entre 0 e 60 mil pra desenhar a reta
plt.plot(X, t0 + t1*X, "b") #reta azul que o modelo aprendeu: satisfação = θ0 + θ1 * PIB
#valores de θ0 e θ1 escritos no gráfico (são os mesmos que o modelo achou, arredondados)
plt.text(5000, 3.1, r"$\theta_0 =4.85$", fontsize=14, color="b")
plt.text(5000, 2.2, r"$\theta_1 =4.91 \times 10^{-5}$",fontsize=14,color="b")
plt.show() #abre a janela do gráfico