#VERSÃO DO PYTHON NECESSÁRIA
import sys
assert sys.version_info >= (3, 5)

# VERSÃO DO SCIKIT-LEARN :> = 0.20
import sklearn
assert sklearn.__version__ >= "0.20"

#TREINAMENTO E EXECUÇÃO DE UM MODELO LINEAR USANDO SICKIT LEARN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# Esta função apenas combina os dados de satisfação com a vida da OCDE e os dados de PIB per capita do FMI.
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    #remover indices de paises que não são relevantes para o modelo
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
