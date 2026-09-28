from p15_atributos_texto_e_transformadores import housing, housing_labels, full_pipeline
from p01_configuracao_inicial import save_fig
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import geom, expon
import matplotlib.pyplot as plt
import joblib
import os

#material extra do notebook (não está no livro impresso)

#um pipeline completo com preparação e previsão juntas
#o pipeline do p15 vira a primeira etapa, e o modelo entra como ultima etapa
#assim o pipeline recebe os dados crus (com texto e valores vazios) e já devolve o preço previsto
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

if __name__ == "__main__":
    #o fit prepara os dados e treina o modelo de uma vez
    full_pipeline_with_predictor.fit(housing, housing_labels)
    some_data = housing.iloc[:5]
    #as mesmas previsões da regressão linear do p16, mas sem precisar chamar o full_pipeline.transform antes
    print(full_pipeline_with_predictor.predict(some_data))

    #persistencia do modelo usando joblib
    #salvar o modelo treinado num arquivo, para usar depois sem precisar treinar de novo
    #a pasta modelos/ fica no .gitignore, porque o arquivo é gerado aqui e não precisa ir para o GitHub
    my_model = full_pipeline_with_predictor
    os.makedirs("modelos", exist_ok=True)
    caminho = os.path.join("modelos", "my_model.pkl")
    joblib.dump(my_model, caminho)#salva
    #...
    my_model_loaded = joblib.load(caminho)#carrega de volta, já treinado
    #conferindo: o modelo carregado prevê exatamente o mesmo que o original
    print(my_model_loaded.predict(some_data))

    #exemplo de distribuições do SciPy para o RandomizedSearchCV
    #geom (geometrica): sorteia numeros inteiros, o 1 é o mais comum e cada numero seguinte tem metade da chance
    #expon (exponencial): numeros quebrados, os pequenos são mais comuns e os grandes vão ficando raros
    #rvs(10000) sorteia 10000 valores, para ver o formato da distribuição no histograma
    geom_distrib=geom(0.5).rvs(10000, random_state=42)
    expon_distrib=expon(scale=1).rvs(10000, random_state=42)
    plt.hist(geom_distrib, bins=50)
    save_fig("distribuicao_geometrica")
    plt.show()
    plt.hist(expon_distrib, bins=50)
    save_fig("distribuicao_expon")
    plt.show()
