from p02_baixar_dados import load_housing_data
from p06_conjunto_teste_por_id import split_train_test_by_id

housing = load_housing_data()

#o housing não tem coluna de id, então o livro cria uma
#opção A: usar o numero da linha como id
#só funciona se os dados novos forem sempre colocados no final e nenhuma linha for apagada
housing_with_id = housing.reset_index() #adiciona coluna index
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(len(test_set))
print(len(train_set))
