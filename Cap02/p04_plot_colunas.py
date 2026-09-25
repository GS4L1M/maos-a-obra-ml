from p02_baixar_dados import  load_housing_data
import matplotlib.pyplot as plt
from p01_configuracao_inicial import save_fig
housing = load_housing_data()
housing.hist(bins=50, figsize=(20,15))
save_fig("atributos_histograma")
plt.show()