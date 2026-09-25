# Capítulo 2: Projeto de ML de ponta a ponta

Neste capítulo o livro usa a base **California Housing**: 20.640 distritos da Califórnia, com dados do censo de 1990 (renda, população, idade das casas, localização etc.). O objetivo é prever o `median_house_value`, ou seja, o valor mediano das casas de cada distrito.

## Ordem dos arquivos

Os arquivos têm um prefixo `p01`, `p02`... para aparecerem na ordem em que o livro faz as coisas. O `p` vem antes do número porque o Python não aceita importar um arquivo cujo nome começa com número (`from 02_baixar_dados import ...` dá `SyntaxError`).

| Arquivo | O que faz |
|---|---|
| `p01_configuracao_inicial.py` | confere as versões, configura o matplotlib e cria a função `save_fig` |
| `p02_baixar_dados.py` | baixa a base (se faltar) e tem a função `load_housing_data` |
| `p03_visualizando_dados.py` | primeira olhada nos dados (`head`, `info`, `value_counts`) |
| `p04_plot_colunas.py` | histograma de todas as colunas numéricas |
| `p05_conjunto_teste_aleatorio.py` | conjunto de teste por sorteio feito na mão |
| `p06_conjunto_teste_por_id.py` | só as funções para separar pelo hash do id |
| `p07_conjunto_teste_id_indice.py` | usa o p06 com o número da linha como id |
| `p08_conjunto_teste_id_localizacao.py` | usa o p06 com a longitude e a latitude como id |
| `p09_conjunto_teste_sklearn.py` | conjunto de teste com o `train_test_split` |
| `p10_conjunto_teste_estratificado.py` | conjunto de teste estratificado por faixa de renda (o que o livro usa) |

Rode sempre a partir da pasta raiz do projeto: `python Cap02/p04_plot_colunas.py`.

## Criando o conjunto de teste

### O problema

Antes de olhar os dados com calma, é preciso separar uma parte deles (20%) para o **teste**, e depois não mexer mais nela. Se eu ficar olhando o teste, meu cérebro acaba escolhendo o modelo "pensando" nele. Quando eu for medir o resultado, o número vai sair otimista demais. O livro chama isso de *data snooping bias*.

O teste precisa ter duas qualidades:
1. **ser sempre o mesmo**, não importa quantas vezes eu rode o código;
2. **representar bem a base inteira**, principalmente naquilo que mais importa para prever o preço.

### Os jeitos que o livro mostra (e por que não ficar com eles)

| Jeito | Arquivo | Resolve o 1? | Resolve o 2? |
|---|---|---|---|
| Sorteio na mão (`np.random.permutation`) | `p05_conjunto_teste_aleatorio.py` | Só com `seed`, e só se os dados não mudarem | Não |
| Hash do id (`crc32`) | `p06` + `p07` / `p08` | Sim, até com dados novos | Não |
| `train_test_split` do sklearn | `p09_conjunto_teste_sklearn.py` | Sim (`random_state`) | Não |
| **Amostragem estratificada** | `p10_conjunto_teste_estratificado.py` | **Sim** | **Sim** |

Os três primeiros sorteiam as casas "no escuro". Numa base grande isso até funciona, mas pode sair um teste com, por exemplo, poucas casas de renda alta. É como fazer uma pesquisa de opinião e, sem querer, entrevistar quase só homens: o resultado fica torto. O livro chama isso de **viés de amostragem**.

### O melhor jeito: amostragem estratificada

A ideia é: **a renda (`median_income`) é o que mais ajuda a prever o preço das casas**, então o teste precisa ter a mesma proporção de rendas baixas, médias e altas que a base inteira. Para isso, eu divido as casas em faixas de renda (os **estratos**) e sorteio 20% de **cada faixa**.

### Ferramentas

| Ferramenta | De onde vem | Para que serve aqui |
|---|---|---|
| `load_housing_data()` | `p02_baixar_dados.py` | carregar a base |
| `pd.cut()` | pandas | transformar a renda (número contínuo) em 5 faixas |
| `np.inf` | numpy | "infinito", o limite de cima da última faixa |
| `StratifiedShuffleSplit` | `sklearn.model_selection` | sortear respeitando a proporção de cada faixa |
| `.loc[]` | pandas | pegar as linhas a partir dos índices sorteados |
| `.value_counts() / len()` | pandas | conferir as proporções de cada faixa |
| `.drop()` | pandas | apagar a coluna de faixas no final |

### Passo a passo

Sugestão de arquivo: `p10_conjunto_teste_estratificado.py`.

1. **Carregar a base** com o `load_housing_data()`.

2. **Criar a coluna `income_cat`** com o `pd.cut()`. A renda está em dezenas de milhares de dólares (3.5 = US$ 35 mil). O livro usa estes limites:
   - `bins=[0., 1.5, 3.0, 4.5, 6., np.inf]`
   - `labels=[1, 2, 3, 4, 5]`

   Isso cria 5 faixas: de 0 a 1.5 é a faixa 1, de 1.5 a 3.0 é a faixa 2, e assim por diante, até a faixa 5, que vai de 6 para cima.

3. **Conferir as faixas** com `value_counts()`. Um histograma (`.hist()`) também ajuda a enxergar.

4. **Separar com o `StratifiedShuffleSplit`.** Ele funciona em duas etapas:
   - **criar** o separador com `n_splits=1` (quero só uma divisão), `test_size=0.2` e `random_state=42`;
   - **usar** o separador com `.split(dados, coluna_das_faixas)`. Ele devolve os **índices** de treino e de teste dentro de um `for`, e com eles eu pego as linhas usando `housing.loc[...]`.

   Os nomes que o livro usa são `strat_train_set` e `strat_test_set`.

5. **Conferir se funcionou:** calcular a proporção de cada faixa no `strat_test_set` e comparar com a da base inteira. Os números têm que sair quase iguais.

6. **Apagar a coluna `income_cat`** dos dois conjuntos com `.drop("income_cat", axis=1)`. Ela só servia para fazer a divisão, e não é um dado real das casas.

### Resultados esperados (para conferir)

Casas em cada faixa (passo 3):

| Faixa | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Casas | 822 | 6581 | 7236 | 3639 | 2362 |

Tamanhos (passo 4): **16512** no treino e **4128** no teste.

Proporções no teste (passo 5):

| Faixa | Base inteira | Estratificado | Sorteio comum (`train_test_split`) |
|---|---|---|---|
| 1 | 3,98% | 4,00% | 4,02% |
| 2 | 31,88% | 31,88% | 32,44% |
| 3 | 35,06% | 35,05% | 35,85% |
| 4 | 17,63% | 17,64% | 16,74% |
| 5 | 11,44% | 11,43% | 10,95% |

Dá para ver que o estratificado fica praticamente igual à base, enquanto o sorteio comum erra por quase um ponto nas faixas 3 e 4.

### Dica de módulos

O resto do capítulo inteiro usa o `strat_train_set`. Então vale deixar a separação dentro de uma **função** que devolve os dois conjuntos (com `return`) e colocar os `print` de conferência dentro de `if __name__ == "__main__":`. Assim, os próximos arquivos só precisam importar a função, sem repetir nada e sem imprimir as conferências toda vez.
