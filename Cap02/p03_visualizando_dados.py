#importando o pandas e carregando o dataset
import pandas as pd
from p02_baixar_dados import  load_housing_data
housing = load_housing_data()
#no .py precisa do print pra aparecer no terminal, no jupyter aparece sozinho
print(housing.head(10))#imprimir as 10 primeiras linhas do dataset

#verificar informações do dataset
#mostra as colunas, o tipo de cada uma e quantos valores não vazios tem
#o total_bedrooms tem 20433 em vez de 20640, ou seja, tem 207 casas sem esse dado
#o info já imprime sozinho, por isso não tem print
housing.info()

# contando os valores casas proximos ao oceano
#o ocean_proximity é a unica coluna de texto, o value_counts conta quantas casas tem em cada categoria
print(housing["ocean_proximity"].value_counts())

# descrição completa do banco de dados
#media, desvio padrão (std), minimo, maximo e os percentis 25%, 50% (mediana) e 75% de cada coluna numerica
print(housing.describe())
