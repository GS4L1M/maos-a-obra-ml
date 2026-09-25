import os 
import tarfile
import urllib.request

#faça o download do dataset caso não tenha
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path): #verifica se já existe no diretorio
        os.makedirs(housing_path) #caso não ele cria o diretorio
    tgz_path = os.path.join(housing_path, "housing.tgz")#faz a extração do arquivo .tgz caso esteja usando um linux
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

    fetch_housing_data() 

#importando o pandas e carregando o dataset
import pandas as pd

 #carrega os dados que fizemos na variavel acima
def load_housing_data(housing_path=HOUSING_PATH): 
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#visão rapida dos dados 

housing = load_housing_data()
housing.head(10)#imprimir as 10 primeiras linhas do dataset
