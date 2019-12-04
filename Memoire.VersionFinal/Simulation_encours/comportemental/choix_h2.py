'''
D'apres les grilles de lissage calculés en choix_fentre.py, on n'a pas aucune
doute que le meilleur choix de h1 ( la fentre de lissage pour la variable age )
est h1 = 1 mois. Pour la ecart de taux, s'est pas tres claire, car les valeurs
de R2 sont tres proches et peut dependre du decoupage aleatoire choisi. Dans ce
fichier on fixe h1 = 1 et on calcule R2 pour plusieurs decoupages
aleatoires et plusieurs valeurs de h2
'''

from stock_pel import stock_pel
import pandas as pd
import numpy as np
import time
from non_parametric_regression import Kernel

# ==============================================================================
# choisir h2 pour le taux de cloture.
# ==============================================================================

print('Computing R2 for taux cloture')

t0 = time.time()
X = stock_pel.values[:, 0:2]
encours = stock_pel.values[:, 2]
clotures = stock_pel.values[:, 4]
tx_cloture = clotures / encours

# On fait une grille avec possibles choises de fenetres h. Chaque entree aura
# le coeficient R2 pour le test_data. On choisi la fenetre que maximise le R2.

seed = [125, 586, 879, 456, 987]

grille = pd.DataFrame(index=seed,
                      columns=[x / 200 for x in range(2, 11)])


for seed in grille.index:
    for h2 in grille.columns:
        RS = np.random.RandomState(seed)
        model = Kernel(X=X, Y=tx_cloture, RS=RS)

        grille[h2][seed] = model.R2(h=[1, h2])

grille.to_csv('h2_clotures.csv')

t1 = time.time()

print(f'on a finit avec les clotures en {t1-t0:.2f} secondes \n')
# ==============================================================================
# choisir h2 pour le taux de versement.
# ==============================================================================

print('Computing R2 for taux versement')

stock_pel = stock_pel[stock_pel['age'] < 120]
X = stock_pel.values[:, 0:2]
encours = stock_pel.values[:, 2]
versements = stock_pel.values[:, 3]
tx_versement = versements / encours


for seed in grille.index:
    for h2 in grille.columns:
        RS = np.random.RandomState(seed)
        model = Kernel(X=X, Y=tx_versement, RS=RS)
        grille[h2][seed] = model.R2(h=[1, h2])

grille.to_csv('h2_versements.csv')

t2 = time.time()

print(f'on a finit avec les versements en {t2-t1:.2f} secondes \n')
