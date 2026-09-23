# O código do livro espera que os arquivos de dados estejam no diretório atual. Fiz um pequeno ajuste aqui para buscar os arquivos em `datasets/lifesat`.
#Também estou fazendo a tradução do jupyter notebook
#Alterei as ordens de alguns codigos pela forma como aprendi com meus professores
import os
import urllib.request
import pandas as pd
data_path = os.path.join("datasets", "lifesat", "")

#Download dos datasets (só baixa se o arquivo ainda não existir)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
os.makedirs(data_path, exist_ok=True)
for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
    if not os.path.exists(os.path.join(data_path, filename)):
        print("Baixando dados", filename)
        url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
        urllib.request.urlretrieve(url, os.path.join(data_path, filename))

# Carregando os dados
oecd_bli = pd.read_csv(os.path.join(data_path, "oecd_bli_2015.csv"), thousands=",")
gdp_per_capita = pd.read_csv(os.path.join(data_path, "gdp_per_capita.csv"), thousands=",", delimiter="\t", encoding="latin1", na_values="n/a")

#Visualizando os nomes das colunas do dataframe (só quando este arquivo é executado diretamente, não quando é importado)
if __name__ == "__main__":
    print(oecd_bli.columns)
    print(gdp_per_capita.columns)
