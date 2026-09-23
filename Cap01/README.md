# Satisfação com a vida e PIB per capita

Esta pasta guarda as duas bases usadas no Capítulo 1 do livro. Uma mede a satisfação com a vida em cada país, e a outra traz o PIB per capita. Juntando as duas, dá para ver se países mais ricos tendem a ser mais felizes.

> Os nomes das colunas e os trechos de código continuam em inglês de propósito, porque são exatamente os nomes que aparecem nos arquivos CSV. Se forem traduzidos, o código para de funcionar.

## Satisfação com a vida
### De onde vêm os dados
Estes dados vêm do Better Life Index, um índice de qualidade de vida da OCDE: http://stats.oecd.org/index.aspx?DataSetCode=BLI

### Como os dados estão organizados
O arquivo tem 3.292 linhas e 17 colunas. Cada linha é o valor de um indicador em um país. As colunas mais importantes são `Country` (país), `Indicator` (o que está sendo medido), `INEQUALITY` (o grupo da população) e `Value` (o valor medido).

    Int64Index: 3292 entries, 0 to 3291
    Data columns (total 17 columns):
    ﻿"LOCATION"              3292 non-null object
    Country                  3292 non-null object
    INDICATOR                3292 non-null object
    Indicator                3292 non-null object
    MEASURE                  3292 non-null object
    Measure                  3292 non-null object
    INEQUALITY               3292 non-null object
    Inequality               3292 non-null object
    Unit Code                3292 non-null object
    Unit                     3292 non-null object
    PowerCode Code           3292 non-null int64
    PowerCode                3292 non-null object
    Reference Period Code    0 non-null float64
    Reference Period         0 non-null float64
    Value                    3292 non-null float64
    Flag Codes               1120 non-null object
    Flags                    1120 non-null object
    dtypes: float64(3), int64(1), object(13)
    memory usage: 462.9+ KB

### Como carregar com o pandas
Aqui o arquivo é lido, e só as linhas com `INEQUALITY` igual a `"TOT"` são mantidas, ou seja, o total da população de cada país. Depois o `pivot` transforma a tabela: cada país vira uma linha e cada indicador vira uma coluna.

    >>> life_sat = pd.read_csv("oecd_bli_2015.csv", thousands=',')
    
    >>> life_sat_total = life_sat[life_sat["INEQUALITY"]=="TOT"]
    
    >>> life_sat_total = life_sat_total.pivot(index="Country", columns="Indicator", values="Value")
    
    >>> life_sat_total.info()
    <class 'pandas.core.frame.DataFrame'>
    Index: 37 entries, Australia to United States
    Data columns (total 24 columns):
    Air pollution                                37 non-null float64
    Assault rate                                 37 non-null float64
    Consultation on rule-making                  37 non-null float64
    Dwellings without basic facilities           37 non-null float64
    Educational attainment                       37 non-null float64
    Employees working very long hours            37 non-null float64
    Employment rate                              37 non-null float64
    Homicide rate                                37 non-null float64
    Household net adjusted disposable income     37 non-null float64
    Household net financial wealth               37 non-null float64
    Housing expenditure                          37 non-null float64
    Job security                                 37 non-null float64
    Life expectancy                              37 non-null float64
    Life satisfaction                            37 non-null float64
    Long-term unemployment rate                  37 non-null float64
    Personal earnings                            37 non-null float64
    Quality of support network                   37 non-null float64
    Rooms per person                             37 non-null float64
    Self-reported health                         37 non-null float64
    Student skills                               37 non-null float64
    Time devoted to leisure and personal care    37 non-null float64
    Voter turnout                                37 non-null float64
    Water quality                                37 non-null float64
    Years in education                           37 non-null float64
    dtypes: float64(24)
    memory usage: 7.2+ KB

### O que cada indicador significa
Depois do `pivot`, sobram 24 indicadores. O que o livro usa é o `Life satisfaction`.

| Indicador (coluna) | Tradução |
|---|---|
| Air pollution | Poluição do ar |
| Assault rate | Taxa de agressões |
| Consultation on rule-making | Participação da população na criação de leis |
| Dwellings without basic facilities | Moradias sem saneamento básico |
| Educational attainment | Nível de escolaridade |
| Employees working very long hours | Trabalhadores com jornadas muito longas |
| Employment rate | Taxa de emprego |
| Homicide rate | Taxa de homicídios |
| Household net adjusted disposable income | Renda familiar disponível líquida ajustada |
| Household net financial wealth | Patrimônio financeiro líquido das famílias |
| Housing expenditure | Gastos com moradia |
| Job security | Segurança no emprego |
| Life expectancy | Expectativa de vida |
| Life satisfaction | Satisfação com a vida |
| Long-term unemployment rate | Taxa de desemprego de longa duração |
| Personal earnings | Rendimentos pessoais |
| Quality of support network | Qualidade da rede de apoio (família e amigos) |
| Rooms per person | Cômodos por pessoa |
| Self-reported health | Saúde autodeclarada |
| Student skills | Habilidades dos estudantes |
| Time devoted to leisure and personal care | Tempo dedicado ao lazer e aos cuidados pessoais |
| Voter turnout | Participação eleitoral |
| Water quality | Qualidade da água |
| Years in education | Anos de estudo |

## PIB per capita
### De onde vêm os dados
Estes dados vêm do site do FMI (Fundo Monetário Internacional): http://goo.gl/j1MSKe

### Como os dados estão organizados
O arquivo tem 190 linhas, uma por país, e 7 colunas. O valor que interessa está na coluna `2015`, que é o PIB per capita daquele ano, em dólares.

    Int64Index: 190 entries, 0 to 189
    Data columns (total 7 columns):
    Country                          190 non-null object
    Subject Descriptor               189 non-null object
    Units                            189 non-null object
    Scale                            189 non-null object
    Country/Series-specific Notes    188 non-null object
    2015                             187 non-null float64
    Estimates Start After            188 non-null float64
    dtypes: float64(2), object(5)
    memory usage: 11.9+ KB

### Como carregar com o pandas
Este arquivo precisa de alguns parâmetros a mais. As colunas são separadas por TAB (`delimiter='\t'`), o arquivo não está em UTF-8 (`encoding='latin1'`) e os valores ausentes aparecem escritos como `n/a`. Depois da leitura, a coluna `2015` é renomeada para `GDP per capita` para ficar mais fácil de entender.

    >>> gdp_per_capita = pd.read_csv(
    ...     datapath+"gdp_per_capita.csv", thousands=',', delimiter='\t',
    ...     encoding='latin1', na_values="n/a", index_col="Country")
    ...
    >>> gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)

