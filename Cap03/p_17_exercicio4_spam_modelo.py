#EXERCÍCIO 4 (parte 2): classificador de spam, transformando os e-mails em números e treinando
#precisa do nltk e do urlextract (pip install nltk urlextract); sem eles, o código pula essas etapas
from p_16_exercicio4_spam_dados import X_train, X_test, y_train, y_test, email_to_text
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np
import re

#stemming: reduz as palavras ao radical ("Computing" e "Computed" viram "comput")
try:
    import nltk

    stemmer = nltk.PorterStemmer()
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None

#o urlextract acha os endereços de site no texto, para trocar todos pela palavra URL
try:
    import urlextract # pode precisar de conexão com a Internet para baixar os nomes de domínio raiz

    url_extractor = urlextract.URLExtract()
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None


#transformador próprio (igual ao CombinedAttributesAdder do Cap02): transforma cada e-mail
#num Counter, que é um dicionário {palavra: quantas vezes aparece}
#cada opção do __init__ liga ou desliga uma etapa, para dar para testar depois qual ajuda
class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                #as URLs mais compridas são trocadas primeiro, para não sobrar pedaço delas
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            #split() separa o texto nos espaços e o Counter conta cada palavra
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

#segundo transformador: transforma as contagens em vetores de números
#o fit monta o vocabulário (as palavras mais comuns do treino) e o transform usa ele
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                #o min(count, 10) evita que um e-mail que repete muito uma palavra domine a contagem
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        #cada palavra ganha um número de coluna; a coluna 0 fica para as palavras fora do vocabulário
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        #matriz esparsa: guarda só os valores diferentes de zero, economizando memória
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


if __name__ == "__main__":
    #testando o stemmer e o extrator de URLs
    if stemmer is not None:
        for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
            print(word, "=>", stemmer.stem(word))
    if url_extractor is not None:
        print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))

    #testando os transformadores em 3 e-mails
    X_few = X_train[:3]
    X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
    print(X_few_wordcounts)

    #vocabulário pequeno (10 palavras) só para dar para ler a matriz
    vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
    X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
    #1ª coluna = palavras fora do vocabulário, as outras = quantas vezes cada palavra do vocabulário aparece
    print(X_few_vectors.toarray())
    print(vocab_transformer.vocabulary_)

    #juntando os dois transformadores num pipeline e aplicando no treino todo
    preprocess_pipeline = Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer()),
    ])

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)

    #primeiro classificador de spam: regressão logística (mais de 98,5% de acurácia)
    log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
    print(f"acurácia média na validação cruzada: {score.mean()}")

    #precisão e revocação no conjunto de teste
    X_test_transformed = preprocess_pipeline.transform(X_test)

    log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    log_clf.fit(X_train_transformed, y_train)

    y_pred = log_clf.predict(X_test_transformed)

    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
