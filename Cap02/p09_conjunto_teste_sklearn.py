from p02_baixar_dados import load_housing_data
from sklearn.model_selection import train_test_split

housing = load_housing_data()

#jeito 3: o sklearn já faz a separação pronta, igual ao split_train_test do p05_conjunto_teste_aleatorio
#test_size=0.2 é a porcentagem do teste e o random_state=42 faz o sorteio sair sempre igual
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(test_set.head())
