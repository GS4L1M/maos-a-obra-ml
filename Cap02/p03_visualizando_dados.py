#importando o pandas e carregando o dataset
import pandas as pd
from p02_baixar_dados import  load_housing_data
housing = load_housing_data()
print(housing.head(10))#imprimir as 10 primeiras linhas do dataset

#verificar informações do dataset 
housing.info()

# contando os valores casas proximos ao oceano 
print(housing["ocean_proximity"].value_counts())

# descrição completa do banco de dados
print(housing.describe())