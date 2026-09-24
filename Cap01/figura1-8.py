#Grafico mostrando a regularização: deixar o modelo mais "simples" pra ele não exagerar
#comparando 3 retas: todos os dados, dados parciais e dados parciais com regularização
#sample_data = os 29 países / missing_data = os 7 que tiramos / status_de_todos_paises = os 36
from manipulando_dados import sample_data, missing_data, status_de_todos_paises
import numpy as np #contas com vetores
import matplotlib.pyplot as plt #gráficos
from sklearn import linear_model #modelos lineares do scikit-learn

#modelo com os 29 países (o mesmo da figura1-3)
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
lin1 = linear_model.LinearRegression()
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]

#modelo com todos os 36 países (o mesmo da figura1-6)
Xfull = np.c_[status_de_todos_paises["GDP per capita"]]
yfull = np.c_[status_de_todos_paises["Life satisfaction"]]
lin_reg_full = linear_model.LinearRegression()
lin_reg_full.fit(Xfull, yfull)
t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]

plt.figure(figsize=(8,3)) #cria a figura vazia, dessa vez sem o .plot do pandas
plt.xlabel("GDP per capita")
plt.ylabel('Life satisfaction')

#pontos: azul ("bo") = os 29 países usados no treino / quadrado vermelho ("rs") = os 7 de fora
plt.plot(list(sample_data["GDP per capita"]), list(sample_data["Life satisfaction"]), "bo")
plt.plot(list(missing_data["GDP per capita"]), list(missing_data["Life satisfaction"]), "rs")

X = np.linspace(0, 110000, 1000) #1000 valores de PIB entre 0 e 110 mil
#label é o texto que aparece na legenda
plt.plot(X, t0full + t1full * X, "r--", label="Linear model on all data") #vermelha tracejada: todos os dados
plt.plot(X, t0 + t1*X, "b:", label="Linear model on partial data") #azul pontilhada: só os 29 países

#Ridge é uma regressão linear com regularização: o alpha controla o quanto ela "segura" a inclinação
#alpha bem alto (10^9.5) deixa a reta mais deitada, mesmo treinando só com os 29 países
ridge = linear_model.Ridge(alpha=10**9.5)
ridge.fit(Xsample, ysample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0] #θ0 e θ1 do modelo regularizado
plt.plot(X, t0ridge + t1ridge * X, "b", label="Regularized linear model on partial data") #azul contínua

#a reta regularizada fica mais perto da vermelha (todos os dados), mesmo sem ter visto os 7 países de fora
plt.legend(loc="lower right") #mostra a legenda no canto de baixo à direita
plt.axis([0, 110000, 0, 10])
plt.xlabel("GDP per capita (USD)")
plt.show()
