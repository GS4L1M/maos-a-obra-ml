# É necessário Python ≥3.5
import sys
assert sys.version_info >= (3, 5)

# É necessário Scikit-Learn ≥0.20
import sklearn
assert sklearn.__version__ >= "0.20"

# Imports comuns
import numpy as np
import os

# para deixar a saída deste notebook igual em todas as execuções
np.random.seed(42)

# Para plotar figuras bonitas
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Onde salvar as figuras
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

#salvar as figuras que iremos criar
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

