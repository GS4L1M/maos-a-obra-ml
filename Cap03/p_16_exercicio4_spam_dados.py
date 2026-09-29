#EXERCÍCIO 4 (parte 1): classificador de spam, baixando e preparando os e-mails
#o p17 importa daqui os conjuntos de treino/teste e a função email_to_text
from sklearn.model_selection import train_test_split
from collections import Counter
from html import unescape
import numpy as np
import urllib.request
import tarfile
import email
import email.policy
import re
import os

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
#os e-mails ficam em datasets/spam, na raiz do projeto
SPAM_PATH = os.path.join("datasets", "spam")

#baixa os dois arquivos compactados (ham = e-mail normal, spam = lixo) e descompacta
def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        #o filter="data" não está no livro; ele impede que um arquivo compactado grave fora da pasta
        tar_bz2_file.extractall(path=spam_path, filter="data")
        tar_bz2_file.close()

#o módulo email do Python lê o arquivo e já separa cabeçalho, corpo, codificação etc.
def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

#descreve o formato do e-mail; se ele tem várias partes (multipart), olha cada parte
#a função chama ela mesma para as partes de dentro (isso se chama recursão)
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

#conta quantos e-mails existem de cada formato
def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

#transforma HTML em texto usando expressões regulares (re.sub troca o que casa com o padrão):
#tira o <head>, troca os links <a> pela palavra HYPERLINK, apaga as outras tags,
#junta linhas em branco seguidas e converte coisas como &gt; de volta para >
def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

#devolve o conteúdo do e-mail em texto puro, seja qual for o formato
#se tiver parte em texto, usa ela; se só tiver HTML, converte com a função de cima
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # caso haja problemas de codificação
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
#o len(name) > 20 pula o arquivo "cmds" que vem junto e não é e-mail
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

#separando treino e teste antes de olhar demais os dados
#y = 0 para ham e 1 para spam; dtype=object porque cada item é um e-mail, e não um número
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    print(f"quantidade de ham: {len(ham_filenames)}")
    print(f"quantidade de spam: {len(spam_filenames)}")

    #um exemplo de cada, para ver como são os dados
    print(ham_emails[1].get_content().strip())
    print(spam_emails[6].get_content().strip())

    #ham costuma ser texto puro e às vezes assinado com PGP; spam tem muito HTML
    print(structures_counter(ham_emails).most_common())
    print(structures_counter(spam_emails).most_common())

    #os cabeçalhos de um spam; o livro vai usar só o assunto (Subject)
    for header, value in spam_emails[0].items():
        print(header,":",value)
    print(spam_emails[0]["Subject"])

    #testando o html_to_plain_text num spam em HTML
    html_spam_emails = [email for email in X_train[y_train==1]
                        if get_email_structure(email) == "text/html"]
    sample_html_spam = html_spam_emails[7]
    print(sample_html_spam.get_content().strip()[:1000], "...")
    print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
    print(email_to_text(sample_html_spam)[:100], "...")
