from p02_baixar_dados import load_housing_data
import numpy as np

#apenas para demonstrar o train_test_split

#jeito 1: separa treino e teste por sorteio
#embaralha os numeros das linhas, pega os primeiros 20% pro teste e o resto pro treino
#problema: se o dataset ganhar linhas novas o sorteio muda e casas do teste podem ir pro treino
def split_train_test(data, test_ratio):
    shufled_indices = np.random.permutation(len(data))#lista dos numeros das linhas embaralhada
    test_set_size = int(len(data) * test_ratio)#quantas linhas vão pro teste (20% de 20640 = 4128)
    test_indices = shufled_indices[:test_set_size]#do começo até 4128 = teste
    train_indices = shufled_indices[test_set_size:]#de 4128 até o fim = treino
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    housing = load_housing_data()
    # torna a saida identica em todas as execuções
    np.random.seed(42)
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(test_set))
    print(len(train_set))
