# Mãos à Obra: Aprendizado de Máquina

Aqui eu tô guardando meus estudos do livro **"Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow"**, do Aurélien Géron.

Peguei os notebooks originais do livro ([ageron/handson-ml2](https://github.com/ageron/handson-ml2)) e passei tudo de Jupyter Notebook pra scripts `.py`. Também fui traduzindo pro português e quebrando o código em arquivos menores. Mudei a ordem de alguns trechos pra ficar do jeito que aprendi com meus professores.

## 📖 Dedicatória e créditos

Esse repositório só existe por causa desse livro, então nada mais justo que dar os créditos pra quem merece.

**Obrigado, Aurélien Géron!** Seu livro é uma das melhores portas de entrada pro Machine Learning. Ele explica a teoria sem enrolação e bota a gente pra colocar a mão na massa desde o primeiro capítulo. Todo o conteúdo, os exemplos e o código original são trabalho seu. Aqui eu só tô estudando, adaptando e anotando o que aprendo pelo caminho.

- **Autor:** Aurélien Géron
- **Livro original:** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2ª edição), publicado pela **O'Reilly Media**
- **Edição brasileira:** *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*, publicada pela **Alta Books**
- **Código original:** [github.com/ageron/handson-ml2](https://github.com/ageron/handson-ml2), com licença Apache 2.0
- **Dados:** OCDE (Better Life Index) e FMI (PIB per capita)

Se você curte ML, compra o livro! Vale cada página. 🙌

> Esse repositório é só pra estudo pessoal e não tem nenhuma ligação oficial com o autor nem com as editoras.

## 📁 Como tá organizado

```
Maos a obra aprendizado de maquinas/
├── Cap01/
│   ├── carregar_dados.py    # baixa os CSVs (se ainda não tiver) e carrega os dados da OCDE e do FMI
│   ├── combinar_dados.py    # função prepare_country_stats, que junta as duas bases
│   └── plotando_grafico.py  # pega os dados, prepara e plota o gráfico
└── datasets/
    └── lifesat/
        ├── oecd_bli_2015.csv    # satisfação com a vida (OCDE)
        └── gdp_per_capita.csv   # PIB per capita (FMI)
```

## ▶️ Como rodar

Roda os scripts **a partir da pasta raiz do projeto**, porque os caminhos `datasets/lifesat/...` partem dela:

```powershell
.venv\Scripts\activate
python Cap01/plotando_grafico.py
```

Vai precisar de: `pandas`, `numpy`, `matplotlib` e `scikit-learn`.

## 🤔 Jupyter Notebook x `.py`: o que muda

Umas coisas que me pegaram quando passei do notebook pro `.py`:

| Situação | No Jupyter | No `.py` |
|---|---|---|
| Mostrar gráfico | Aparece sozinho no fim da célula | Tem que chamar `mpl.show()`, que abre uma janela |
| Usar variável de outra parte do código | Todas as células dividem a mesma memória | Cada arquivo é separado, então tem que **importar** o que vem de outro arquivo |

Por isso o `plotando_grafico.py` importa os dados e a função dos outros arquivos:

```python
from carregar_dados import oecd_bli, gdp_per_capita
from combinar_dados import prepare_country_stats
```

Detalhe: no `import` não vai o `.py` no nome do arquivo.

**Dica pra quem sente falta do Jupyter:** instalando as extensões *Python* e *Jupyter* no VS Code, dá pra dividir o `.py` em células com `# %%` e clicar em **Run Cell**. Os gráficos aparecem na *Interactive Window*, igualzinho ao notebook, sem precisar de `show()`.

## 🛠️ O que eu fui mudando

### Cap01: `carregar_dados.py`
- Ajustei os caminhos pra buscar os arquivos em `datasets/lifesat/`, porque o livro espera os CSVs na mesma pasta do código.
- Trouxe o **download** dos dados do `plotando_grafico.py` pra cá. Agora ele só baixa um CSV se o arquivo ainda não estiver na pasta.
- Agora ele carrega **as duas bases**. O `gdp_per_capita.csv` é meio chatinho e precisa de uns parâmetros a mais:
  ```python
  gdp_per_capita = pd.read_csv(os.path.join(data_path, "gdp_per_capita.csv"),
                               thousands=",", delimiter="\t",
                               encoding="latin1", na_values="n/a")
  ```
  - `delimiter="\t"`: as colunas são separadas por TAB e não por vírgula.
  - `encoding="latin1"`: o arquivo não tá em UTF-8.
  - `na_values="n/a"`: onde tiver `n/a`, o pandas entende como valor vazio.
- Coloquei os `print` das colunas dentro de `if __name__ == "__main__":`. Assim eles só aparecem quando eu rodo esse arquivo direto, e não quando outro arquivo importa ele.

### Cap01: `combinar_dados.py`
- **Erro `KeyError: 'country'`:** os nomes das colunas estavam em minúsculo (`"country"`, `"indicator"`, `"value"`), mas nos CSVs eles começam com maiúscula. Consertei:
  ```python
  oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
  gdp_per_capita.set_index("Country", inplace=True)
  ```

### Cap01: `plotando_grafico.py`
- **Erro `NameError: name 'prepare_country_stats' is not defined`:** faltava importar a função. Coloquei `from combinar_dados import prepare_country_stats`.
- **Nomes de coluna errados:** estava `"life satisfaction"` e `'life satisfacition'` (com erro de digitação). O certo é `"Life satisfaction"`, e o pandas diferencia maiúscula de minúscula.
- **`plt.show()` virou `mpl.show()`:** nesse arquivo eu importei o matplotlib como `mpl`, não como `plt`.
- **Organização:** tirei o download e a leitura dos CSVs daqui. Agora ele só importa os dados prontos com `from carregar_dados import oecd_bli, gdp_per_capita`. Também apaguei os imports que ficaram sobrando (`pandas`, `os`, `urllib.request`).
