from p02_baixar_dados import  load_housing_data
import matplotlib.pyplot as plt
from p01_configuracao_inicial import save_fig
housing = load_housing_data()
#histograma de todas as colunas numericas de uma vez
#o histograma mostra quantas casas tem em cada faixa de valor
#bins=50 divide cada grafico em 50 barrinhas e figsize é o tamanho da figura
#no livro dá pra ver que o median_house_value e o housing_median_age têm um "teto" (a barra alta no final)
housing.hist(bins=50, figsize=(20,15))
save_fig("atributos_histograma")
plt.show()
