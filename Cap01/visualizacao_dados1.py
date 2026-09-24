# importando pandas para visualização dos dados
import pandas as pd

#Carregando os dados com o modulo que criamos
from carregar_dados import oecd_bli, gdp_per_capita
#Visualizando o cabeçalho do dataframe
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
pd.set_option("display.max_columns", None)

#imprimindo as 10 primeiras linhas do dataframe
print(oecd_bli.head(10))

#imprimindo a coluna de satisfação com a vida
print(oecd_bli["Life satisfaction"].head())

#imprimindo o cabeçalho do dataframe do PIB per capita
print(gdp_per_capita.head(10))

#Juntado os dados para verificar a satisfação de vida nos EUA
from combinar_dados import juntar_dados
status_de_todos_paises = juntar_dados(oecd_bli, gdp_per_capita)
print(status_de_todos_paises[["GDP per capita", "Life satisfaction"]].loc["United States"])

