#Grafico mostrando que os dados de treino precisam representar bem os casos novos
#a reta pontilhada foi treinada só com os 29 países e a reta preta com todos os 36
#tabelas prontas do manipulando_dados:
#sample_data = os 29 países / missing_data = os 7 que tiramos / status_de_todos_paises = os 36
from manipulando_dados import sample_data, missing_data, status_de_todos_paises
import numpy as np #contas com vetores
import matplotlib.pyplot as plt #gráficos
from sklearn import linear_model #modelos lineares do scikit-learn

#treinando o modelo só com os 29 países (o mesmo da figura1-3)
lin1 = linear_model.LinearRegression()
Xsample = np.c_[sample_data["GDP per capita"]]
ysample = np.c_[sample_data["Life satisfaction"]]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0] #θ0 e θ1 da reta com dados parciais

#posição do nome de cada país que tinha ficado de fora (escolhida na mão, igual à figura1-1)
position_text2 = {
    "Brazil": (1000, 9.0),
    "Mexico": (11000, 9.0),
    "Chile": (25000, 9.0),
    "Czech Republic": (35000, 9.0),
    "Norway": (60000, 3),
    "Switzerland": (72000, 3.0),
    "Luxembourg": (90000, 3.0),
}

#pontos dos 29 países, agora com o eixo x até 110 mil pra caber o Luxemburgo
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(8,3))
plt.axis([0, 110000, 0, 10])

#para cada país que tinha ficado de fora: escreve o nome com seta e marca com quadrado vermelho ("rs")
for country, pos_text in position_text2.items():
    pos_data_x, pos_data_y = missing_data.loc[country] #PIB e satisfação do país
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")

X=np.linspace(0, 110000, 1000) #1000 valores de PIB entre 0 e 110 mil
plt.plot(X, t0 + t1*X, "b:") #reta azul pontilhada: modelo treinado só com os 29 países

#treinando outro modelo agora com TODOS os 36 países
lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[status_de_todos_paises["GDP per capita"]]
yfull = np.c_[status_de_todos_paises["Life satisfaction"]]
lin_reg_full.fit(Xfull, yfull)

t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0] #θ0 e θ1 com todos os dados
X = np.linspace(0, 110000, 1000)
plt.plot(X, t0full + t1full * X, "k") #reta preta: modelo com todos os países ("k" = preto)
plt.xlabel("GDP per capita (USD)")

#as duas retas ficam bem diferentes: com poucos dados o modelo achava que PIB alto = muito feliz,
#mas os países ricos (Noruega, Suíça, Luxemburgo) não são tão mais felizes assim
plt.show()
