from p15_atributos_texto_e_transformadores import housing, housing_labels, full_pipeline
from p17_ajuste_fino import feature_importances
from p22_exercicio3_selecionar_atributos import TopFeatureSelector, k
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

#Exercicio 4
#Pergunta: tente criar um unico pipeline que faça toda a preparação dos dados mais a previsão final.

#os melhores hiperparametros do SVR que a busca aleatoria do p21 encontrou
#no notebook é SVR(**rnd_search.best_params_), mas importar o rnd_search faria a busca de ~10 minutos rodar de novo
#então copiei os valores que o p21 imprimiu
#o ** "abre" o dicionario: SVR(**{'C': 157055, ...}) é o mesmo que SVR(C=157055, ...)
best_params = {'C': 157055.10989448498, 'gamma': 0.26497040005002437, 'kernel': 'rbf'}

#as 3 etapas num pipeline só: preparação (p15) -> seleção das k melhores colunas (p22) -> modelo
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**best_params))
])

if __name__ == "__main__":
    #o fit recebe os dados crus, com texto e valores vazios, e faz tudo de uma vez
    prepare_select_and_predict_pipeline.fit(housing, housing_labels)

    #testando o pipeline completo em 4 casas
    some_data = housing.iloc[:4]
    some_labels = housing_labels.iloc[:4]
    #o \t é um TAB, só para as duas listas ficarem alinhadas no terminal
    print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
    print("Labels:\t\t", list(some_labels))
    #o pipeline funciona, mas as previsões não são fantasticas
    #ficariam bem melhores com o random forest do p17 no lugar do SVR
