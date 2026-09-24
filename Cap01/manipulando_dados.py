from carregar_dados import oecd_bli, gdp_per_capita
from combinar_dados import juntar_dados, prepare_country_stats
import pandas as pd

pd.set_option("display.max_columns", None)

#Tabela completa: todos os países em comum e todos os indicadores, ordenados pelo PIB
status_de_todos_paises = juntar_dados(oecd_bli, gdp_per_capita)
print(status_de_todos_paises)

#Tabela usada no modelo: só PIB e satisfação, sem os 7 países das pontas
status_para_modelo = prepare_country_stats(oecd_bli, gdp_per_capita)
print(status_para_modelo)
