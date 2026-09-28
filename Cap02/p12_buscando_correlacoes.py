from p10_conjunto_teste_estratificado import strat_train_set
from p01_configuracao_inicial import save_fig
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

housing = strat_train_set.copy()
#buscando correlações
#correlação mede o quanto duas colunas andam juntas, vai de -1 a 1
#perto de 1: quando uma sobe a outra sobe junto / perto de -1: quando uma sobe a outra desce / perto de 0: sem relação em linha reta
# O pandas 20.0 o argumento numeric_only tem como padrão False então é necessário definir como True
#sem ele dá erro por causa da coluna de texto ocean_proximity

corr_matrix = housing.corr(numeric_only=True)#tabela com a correlação de cada coluna com todas as outras
#pega só a coluna do preço e ordena, pra ver quais colunas mais ajudam a prever o preço
corr_matrix["median_house_value"].sort_values(ascending=True)
print(corr_matrix)

#scatter_matrix faz um grafico de dispersão de cada coluna contra cada outra
#com as 11 colunas numericas seriam 121 graficos, por isso o livro escolhe só as 4 mais promissoras
#na diagonal (coluna contra ela mesma) aparece o histograma
attribute = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attribute], figsize=(12, 8))
save_fig("scatter_matrix_plot")
plt.show()

#a renda é o atributo que mais se relaciona com o preço, então o livro olha ela de perto
#plt.axis([x minimo, x maximo, y minimo, y maximo]) define os limites dos eixos
#dá pra ver a linha reta no 500000: é o teto do preço na base, e umas linhas mais fracas no 450000, 350000...
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")
plt.show()
