import numpy as np
import matplotlib.pyplot as plt
from p_1_configuracao_inicial import save_fig

#gerando dados aleatorios para plotarmos a regressão linear
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#criando equação normal para o modelo de regressão linear
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("generated_data_plot")
plt.show()


