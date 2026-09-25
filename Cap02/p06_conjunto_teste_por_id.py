import numpy as np
from zlib import crc32

#jeito 2: separa pelo id de cada casa em vez de sortear
#a mesma casa sempre cai no mesmo lugar, mesmo rodando de novo ou com dados novos
#esse arquivo só tem as funções, quem usa são o p07_conjunto_teste_id_indice e o p08_conjunto_teste_id_localizacao

#responde se uma casa vai pro teste (True) ou não (False)
#o crc32 transforma o id num codigo (hash) entre 0 e ~4,3 bilhões (2**32), sempre o mesmo pro mesmo id
#se o codigo cair nos primeiros 20% dessa faixa a casa vai pro teste
#o & 0xffffffff só garante que o numero fique dentro dessa faixa
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

#usa o test_set_check em todas as casas e separa as duas partes
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]#pega a coluna de ids
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))#roda o check em cada id, vira uma coluna de True/False
    return data.loc[~in_test_set], data.loc[in_test_set]#~ inverte: False = treino, True = teste
