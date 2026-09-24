#Grafico utilizando para demonstrar o apredinzado baseado em modelos usando regressão linear
#pagina 17 do livro 
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

#gráfico de dispersão: um ponto por país, PIB no eixo x e satisfação no eixo y
sample_data.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
#limites dos eixos: x de 0 a 60 mil e y de 0 a 10
plt.axis([0, 60000, 0, 10])

#posição (x, y) onde o nome de cada país vai aparecer, escolhida na mão só pro texto não ficar em cima dos pontos
position_text = {
    "Hungary": (50000, 1), #no livro é (5000, 1)
    "Korea": (18000, 1.7),
    "France": (29000, 2.4),
    "Australia": (40000, 3.8), #no livro é (40000, 3.0)
    #no livro ainda tem "United States": (52000, 3.8)
}

#para cada país do dicionário: pega os valores reais, escreve o nome com uma seta e pinta o ponto de vermelho
for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = sample_data.loc[country] #PIB e satisfação do país
    country = "U.S" if country =="United States" else country #encurta o nome dos EUA
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
                 arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5)) #xy = ponto, xytext = onde fica o texto
    plt.plot(pos_data_x, pos_data_y, "ro") #"ro" = bolinha vermelha
plt.xlabel("GDP per capita") #nome do eixo x
plt.show() #abre a janela do gráfico
