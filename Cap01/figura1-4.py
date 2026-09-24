#Grafico utilizando para demonstrar o apredinzado baseado em modelos usando regressão linear
#importando os dados crus da OCDE e do FMI do nosso modulo carregar_dados
from carregar_dados import oecd_bli, gdp_per_capita
#função que junta as duas tabelas e tira os países das pontas
from combinar_dados import prepare_country_stats, preparar_gdp
#tabelas já prontas do manipulando_dados (sample_data = os 29 países que vão pro gráfico)
#obs: importar esse arquivo roda os print de lá, por isso aparecem tabelas no terminal
from manipulando_dados import remove_indices, keep_indices, sample_data, status_de_todos_paises
import pandas as pd #tabelas
import numpy as np #contas com vetores
import matplotlib.pyplot as plt #gráficos

gdp_preparado = preparar_gdp(gdp_per_capita) #país vira o índice e a coluna 2015 vira GDP per capita
cyprus_gdp_per_capita = gdp_preparado.loc["Cyprus"]["GDP per capita"] #PIB do Chipre
print(cyprus_gdp_per_capita)

from sklearn import linear_model #modelos lineares do scikit-learn
lin1 = linear_model.LinearRegression() #cria o modelo de regressão linear (ainda sem treinar)
#np.c_ transforma a coluna em tabela de 1 coluna, que é o formato que o sklearn espera
Xsample = np.c_[sample_data["GDP per capita"]] #entrada: PIB
ysample = np.c_[sample_data["Life satisfaction"]] #saída: satisfação
lin1.fit(Xsample, ysample) #treina o modelo
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0] #θ0 e θ1 que o modelo achou
#previsão da satisfação do Chipre: [[ ]] porque o predict espera uma tabela (1 país, 1 coluna)
#e o [0][0] tira o número de dentro da resposta, que também vem como tabela
cyprus_predicted_life_satisfaction = lin1.predict([[cyprus_gdp_per_capita]])[0][0]
print(cyprus_predicted_life_satisfaction) #deve dar uns 5.96

#pontos dos 29 países, s=1 deixa as bolinhas bem pequenas pra destacar a previsão
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3), s=1)
plt.xlabel("GDP per capita (USD)") #nome do eixo x
X=np.linspace(0, 60000, 1000) #1000 valores de PIB entre 0 e 60 mil pra desenhar a reta
plt.plot(X, t0 + t1*X, "b") #reta azul que o modelo aprendeu
plt.axis([0, 60000, 0, 10]) #x de 0 a 60 mil e y de 0 a 10
#valores de θ0 e θ1 escritos no gráfico
plt.text(5000, 7.5, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(5000, 6.6, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")
#linha vermelha tracejada ("r--") subindo do PIB do Chipre (y=0) até a reta, onde fica a previsão
plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita], [0, cyprus_predicted_life_satisfaction], "r--")
plt.text(25000, 5.0, r"Prediction = 5.96", fontsize=14, color="b") #texto com o valor previsto
plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro") #bolinha vermelha na previsão
plt.show() #abre a janela do gráfico