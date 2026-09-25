from p02_baixar_dados import load_housing_data
from p06_conjunto_teste_por_id import split_train_test_by_id

housing = load_housing_data()
housing_with_id = housing.reset_index() #adiciona coluna index

#opção B: montar o id com a localização do distrito, que nunca muda
#junta longitude e latitude num numero só
#ex: longitude -122.23 e latitude 37.88 -> -122.23 * 1000 + 37.88 = -122192.12
#o * 1000 empurra a longitude pra longe pra ela não se misturar com a latitude
#distritos no mesmo lugar ficam com o mesmo id, mas tudo bem porque vão juntos pro mesmo conjunto
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
print(test_set.head())
