from p15_atributos_texto_e_transformadores import housing, full_pipeline, housing_prepared
from p17_ajuste_fino import feature_importances, attributes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np

#Exercicio 3
#Pergunta: tente adicionar um transformador no pipeline de preparação para selecionar apenas os atributos mais importantes.

#devolve a posição (indice) dos k maiores valores da lista, em ordem crescente de posição
#argpartition(arr, -k) arruma a lista de um jeito que os k maiores ficam no final, e o [-k:] pega esses k
#ele devolve as posições e não os valores, e o np.sort deixa as posições em ordem (ex: [0, 1, 7, 9, 12])
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

#um transformador igual ao CombineAttributesAdder do p15, mas em vez de criar colunas ele escolhe só as k melhores
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    #aqui o fit aprende alguma coisa: quais são as posições das k colunas mais importantes
    #o _ no final do nome (feature_indices_) é o costume do sklearn para o que foi aprendido no fit
    def fit(self, x, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    #e o transform devolve só essas colunas (todas as linhas, só as colunas escolhidas)
    def transform(self, x):
        return x[:, self.feature_indices_]

#Nota: este seletor usa importancias que já foram calculadas antes (as do random forest do p17)
#daria para calcular dentro do fit, mas a busca em grade do exercicio 5 (p24) ficaria muito lenta,
#porque teria que treinar um random forest a cada combinação de hiperparametros

#numero de atributos que queremos manter
k = 5

#novo pipeline: o pipeline de preparação do p15 + a seleção das k melhores colunas
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

if __name__ == "__main__":
    #as posições das 5 colunas mais importantes
    top_k_feature_indices = indices_of_top_k(feature_importances, k)
    print(top_k_feature_indices)
    #np.array deixa usar a lista de posições para pegar os nomes de uma vez
    #longitude, latitude, median_income, pop_per_hhold e INLAND
    print(np.array(attributes)[top_k_feature_indices])

    #conferindo com a lista ordenada do p17: as 5 primeiras têm que ser as mesmas (só a ordem muda)
    print(sorted(zip(feature_importances, attributes), reverse=True)[:k])

    housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
    #as 3 primeiras casas, agora só com 5 colunas
    print(housing_prepared_top_k_features[0:3])
    #conferindo mais uma vez: pegando as mesmas colunas direto do housing_prepared tem que dar igual
    print(housing_prepared[0:3, top_k_feature_indices])
    #funciona perfeitamente! :)
