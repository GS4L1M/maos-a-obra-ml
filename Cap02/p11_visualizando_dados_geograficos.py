from p10_conjunto_teste_estratificado import strat_train_set
from p01_configuracao_inicial import save_fig, PROJECT_ROOT_DIR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
import urllib.request

#daqui pra frente o livro explora só o treino, pra não "espiar" o teste
#o copy faz uma copia pra poder mexer à vontade sem estragar o strat_train_set original
housing = strat_train_set.copy()

#Visualizando dados greograficos
#Usando o grafico de dispersão scatter
#eixo X os valores de longitude e y latitude
#exemplo de visualização ruim
#os pontos ficam um em cima do outro e não dá pra ver onde tem mais casas
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
plt.show()

#exemplo de uma visualização melhor
#alpha=0.1 deixa cada ponto 90% transparente, onde tem muita casa junta a cor fica mais forte
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
plt.show()

# O argumento sharex=false corrige o bug de exibição, os valores do eixo X e a legenda não estavam sendo exibidos

#s = tamanho do ponto, quanto mais gente no distrito maior o circulo (dividido por 100 pra não ficar gigante)
#c = cor do ponto pelo preço da casa, cmap="jet" vai do azul (barato) ao vermelho (caro)
#colorbar=True mostra a barra de cores do lado
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
plt.show()


# Baixar a imagem da california
#a foto é só um fundo ilustrativo, ela não tem lat e lon dentro, quem desenha a california são os pontos
images_path = os.path.join(PROJECT_ROOT_DIR, "images","end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("baixando arquivo", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename#o + só gruda os textos, por isso precisa da / no final da pasta
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

#criando o grafico com a imagem de california
#mpimg.imread lê o png e transforma numa matriz de pixels
california_img = mpimg.imread(os.path.join(images_path, filename))
#mesmo grafico de cima mas sem a barra de cores automatica (colorbar=False), ela é feita na mão lá embaixo
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                  s=housing["population"]/100, label="population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)
#imshow coloca a foto atrás dos pontos
#extent diz onde esticar a foto: [longitude esquerda, longitude direita, latitude de baixo, latitude de cima]
#esses numeros o autor escolheu pra foto encaixar, dá pra conferir pela fronteira reta com nevada no -120 e no 42
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("latitude", fontsize=14)
plt.xlabel("longitude", fontsize=14)

#barra de cores feita na mão pra mostrar os preços em dolar ($209k, $258k...)
prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)#11 valores espaçados igualmente entre o menor e o maior preço
cbar = plt.colorbar(ticks=tick_values/prices.max())#a barra vai de 0 a 1, por isso divide pelo maior preço
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)#escreve cada valor como $000k
cbar.set_label('Median House Value', fontsize=16)

#resultado: as casas mais caras (vermelho) ficam no litoral, perto de são francisco e los angeles
plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()
