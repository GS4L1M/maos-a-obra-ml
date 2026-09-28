from p10_conjunto_teste_estratificado import strat_train_set
import matplotlib.pyplot as plt
import numpy as np

housing = strat_train_set.copy()
# experimentando com combinações de atributos
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#plotando o grafico coms os valores de quartos por casa e mediana dos valores das casas 
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
plt.axis([0, 5,0,520000])
plt.show()

#DRESCIÇÃO DO DATASET
print(housing.describe())