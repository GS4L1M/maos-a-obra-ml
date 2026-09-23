# EXEMPLO DO CODIGO
import numpy as np
import matplotlib.pyplot as mpl
from carregar_dados import oecd_bli, gdp_per_capita
from combinar_dados import prepare_country_stats
#configurando o tamanho dos gráficos
mpl.rc('axes', labelsize=14) #tamanho do rótulo dos eixos
mpl.rc('xtick', labelsize=12) #tamanho do rótulo do eixo x
mpl.rc('ytick', labelsize=12) #tamanho do rótulo do eixo y

#Preparando os dados para modelo de regressão

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#Visualizando os dados
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
mpl.show()
