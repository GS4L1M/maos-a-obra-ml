from p15_atributos_texto_e_transformadores import housing, housing_labels, full_pipeline
from p17_ajuste_fino import feature_importances
from p23_exercicio4_pipeline_completo import prepare_select_and_predict_pipeline
from sklearn.model_selection import GridSearchCV

#Exercicio 5
#Pergunta: explore automaticamente algumas opções de preparação usando GridSearchCV.

#AVISO: no livro esta busca leva uns 45 minutos. Com n_jobs=-1 fica bem mais rapido, mas ainda demora (uns 11 min aqui).

#Nota: a categoria ISLAND só tem 1 casa no treino, então em alguns folds ela fica de fora da parte de treino
#o notebook resolve com handle_unknown='ignore' no OneHotEncoder (categoria que ele não viu no fit vira tudo 0)
#mas isso não basta aqui: sem a ISLAND no fit, o one-hot cria só 4 colunas em vez de 5,
#a matriz fica com 15 colunas, e o TopFeatureSelector com k=14, 15 ou 16 tenta pegar a coluna 15, que não existe
#(IndexError, e o GridSearchCV marca essas combinações como nan)
#solução: dizer ao OneHotEncoder quais são as 5 categorias, assim ele sempre cria as 5 colunas,
#mesmo quando alguma não aparece no fold. A lista vem do encoder já treinado no p15
categorias = full_pipeline.named_transformers_["cat"].categories_
#set_params muda a configuração do pipeline, e o GridSearchCV copia essa configuração para cada treino
#o nome segue o caminho dentro dos pipelines com __ (dois underlines): preparation -> cat -> categories
prepare_select_and_predict_pipeline.set_params(preparation__cat__categories=categorias)

#o GridSearchCV tambem testa hiperparametros da preparação, não só do modelo, usando o mesmo caminho com __
#estrategia do imputer: media, mediana ou valor mais frequente
#k do seletor: de 1 a 16 colunas (len(feature_importances) = 16)
#3 × 16 = 48 combinações, × 5 folds = 240 treinos
param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

if __name__ == "__main__":
    #o GridSearchCV recebe os dados crus (housing), porque a preparação agora está dentro do pipeline
    grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                    scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search_prep.fit(housing, housing_labels)

    #a melhor estrategia de imputação é most_frequent e quase todas as colunas ajudam (15 de 16)
    #a que fica de fora (ISLAND) parece só adicionar um pouco de ruido
    print(grid_search_prep.best_params_)

    #Parabéns! Você já sabe bastante sobre Machine Learning. :)
