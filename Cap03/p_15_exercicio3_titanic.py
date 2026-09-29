#EXERCÍCIO 3: dataset do Titanic
#prever se um passageiro sobreviveu, com base na idade, sexo, classe, onde embarcou etc.
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import os

#os dados ficam em datasets/titanic, na raiz do projeto (por isso é preciso rodar a partir da raiz)
TITANIC_PATH = os.path.join("datasets", "titanic")
DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/"

#baixa o train.csv e o test.csv, mas só se ainda não estiverem na pasta
def fetch_titanic_data(url=DOWNLOAD_URL, path=TITANIC_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)
    for filename in ("train.csv", "test.csv"):
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            print("Downloading", filename)
            urllib.request.urlretrieve(url + filename, filepath)

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    fetch_titanic_data()
    train_data = load_titanic_data("train.csv")
    #o test.csv não tem a coluna Survived: no Kaggle a resposta fica escondida
    test_data = load_titanic_data("test.csv")

    #Survived é o alvo (0 = não sobreviveu, 1 = sobreviveu)
    #SibSp = irmãos e cônjuges a bordo, Parch = pais e filhos a bordo, Fare = preço pago
    print(train_data.head())

    #o PassengerId vira o índice da tabela
    train_data = train_data.set_index("PassengerId")
    test_data = test_data.set_index("PassengerId")

    #Age, Cabin e Embarked têm valores faltando (menos de 891 não nulos); o Cabin tem 77% nulos
    train_data.info()
    print(f"idade mediana das mulheres: {train_data[train_data['Sex']=='female']['Age'].median()}")

    #só 38% sobreviveram; como é perto de 40%, a acurácia é uma boa métrica aqui
    print(train_data.describe())
    print(train_data["Survived"].value_counts())
    #olhando os atributos categóricos (Embarked: C=Cherbourg, Q=Queenstown, S=Southampton)
    print(train_data["Pclass"].value_counts())
    print(train_data["Sex"].value_counts())
    print(train_data["Embarked"].value_counts())

    #pipeline dos números: preenche o que falta com a mediana e escala
    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    #pipeline das categorias: preenche com o valor mais comum e transforma em colunas de 0 e 1
    #no livro é sparse=False, mas nas versões novas do sklearn o nome mudou para sparse_output
    cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(sparse_output=False)),
        ])

    #Name, Ticket e Cabin ficam de fora por enquanto
    num_attribs = ["Age", "SibSp", "Parch", "Fare"]
    cat_attribs = ["Pclass", "Sex", "Embarked"]

    #aplica cada pipeline nas suas colunas e junta o resultado (igual ao full_pipeline do Cap02)
    preprocess_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])

    X_train = preprocess_pipeline.fit_transform(train_data[num_attribs + cat_attribs])
    print(X_train)
    y_train = train_data["Survived"]

    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_clf.fit(X_train, y_train)

    #no teste só usamos transform (sem fit), para usar as medianas e categorias do treino
    X_test = preprocess_pipeline.transform(test_data[num_attribs + cat_attribs])
    y_pred = forest_clf.predict(X_test)

    #como o teste não tem resposta, a validação cruzada no treino mostra se o modelo é bom
    forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
    print(f"acurácia média do RandomForest: {forest_scores.mean()}")

    svm_clf = SVC(gamma="auto")
    svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
    print(f"acurácia média do SVC: {svm_scores.mean()}")

    #as 10 notas de cada modelo em pontos, com um box plot por cima
    #a caixa vai do quartil de baixo (Q1) ao de cima (Q3); os "bigodes" mostram até onde as notas vão
    #no livro é labels=, mas nas versões novas do matplotlib o nome mudou para tick_labels
    plt.figure(figsize=(8, 4))
    plt.plot([1]*10, svm_scores, ".")
    plt.plot([2]*10, forest_scores, ".")
    plt.boxplot([svm_scores, forest_scores], tick_labels=("SVM","Random Forest"))
    plt.ylabel("Accuracy", fontsize=14)
    plt.show()

    #ideias para melhorar: transformar a idade em faixas de 15 anos
    #(o // divide e joga fora o resto: 37 // 15 * 15 = 30)
    train_data["AgeBucket"] = train_data["Age"] // 15 * 15
    print(train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean())

    #e somar os parentes a bordo (quem viajava sozinho sobreviveu menos)
    train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
    print(train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean())
