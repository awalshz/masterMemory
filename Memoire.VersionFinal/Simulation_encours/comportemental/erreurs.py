'''
Ici on va analyser les erreurs de la regression non parametrique pour 
le taux de cloture et des versements. Le but c'est de trouver sa loi
empirique pour les incluire dans le calcul de la provision, c'est à
dire, en lieu d'utiliser E[Y|X], utiliser E[Y|X] + E, pour chaque taux
de cloture et de versement.
'''

import pandas as pd
from stock_pel import stock_pel
import numpy as np
import time
from non_parametric_regression import Kernel
from nappe import tx_clotures, tx_versements
import matplotlib.pyplot as plt

# ==========================================================================
# Erreus pour le taux de cloture.
# Comme le R2 des erreurs du taux de versements est tres élévé, on ne prendre
# pas en consideration l'erreur de cloture.
# ==========================================================================

print('Computing erreurs pour taux cloture')

t0 = time.time()
X = stock_pel.values[:, 0:2]
encours = stock_pel.values[:, 2]
clotures = stock_pel.values[:, 4]
taux_cloture = clotures / encours

RS = np.random.RandomState(13062013)
model_clotures = Kernel(X=X, Y=taux_cloture, h=[1, 2/100], RS=RS)

E_cloture = model_clotures.Y_test - model_clotures.Y_hat(model_clotures.X_test)

# on fait l'histogramme des erreurs
n, bins, patches = plt.hist(E_cloture, 100, density=True,
                            facecolor='g', alpha=0.75)

plt.xlabel('Erreurs')
plt.ylabel('Probability')
plt.title('Erreurs taux de cloture')
plt.xlim(-0.06, 0.06)
plt.grid(True)
plt.show()
