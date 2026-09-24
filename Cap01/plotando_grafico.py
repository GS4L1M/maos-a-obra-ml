# EXEMPLO DO CODIGO
import numpy as np #contas com vetores
import matplotlib.pyplot as mpl #gráficos (aqui com o apelido mpl)
from carregar_dados import oecd_bli, gdp_per_capita #dados crus da OCDE e do FMI
from combinar_dados import prepare_country_stats #junta as tabelas e tira os países das pontas
#configurando o tamanho dos gráficos
mpl.rc('axes', labelsize=14) #tamanho do rótulo dos eixos
mpl.rc('xtick', labelsize=12) #tamanho do rótulo do eixo x
mpl.rc('ytick', labelsize=12) #tamanho do rótulo do eixo y

#Preparando os dados para modelo de regressão

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita) #29 países com PIB e satisfação
x = np.c_[country_stats["GDP per capita"]] #entrada: PIB
y = np.c_[country_stats["Life satisfaction"]] #saída: satisfação

#Visualizando os dados
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction') #um ponto por país
mpl.show() #abre a janela do gráfico
