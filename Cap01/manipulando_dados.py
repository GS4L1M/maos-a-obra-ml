from carregar_dados import oecd_bli, gdp_per_capita
from combinar_dados import juntar_dados, prepare_country_stats
import pandas as pd

pd.set_option("display.max_columns", None)

#Tabela completa: todos os países em comum e todos os indicadores, ordenados pelo PIB
status_de_todos_paises = juntar_dados(oecd_bli, gdp_per_capita)

#Tabela usada no modelo: só PIB e satisfação, sem os 7 países das pontas
status_para_modelo = prepare_country_stats(oecd_bli, gdp_per_capita)

#remover indices de paises que não são relevantes para o modelo
remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))
sample_data = status_de_todos_paises[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
missing_data = status_de_todos_paises[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

#os print só rodam quando executo este arquivo direto, não quando outro arquivo importa ele
if __name__ == "__main__":
    print(status_de_todos_paises)
    print(status_para_modelo)
