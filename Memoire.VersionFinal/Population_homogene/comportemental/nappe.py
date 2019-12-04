'''
On construit la nappe comportamental, on discretise pour gagner du temps.
De donnes, les ecarts de taux prendre valeurs entre -4% et 3%. On va construir
la nappe entre -5% et 5% a pas 0.5%.

On defini 2 functions tx_clotures(age, ecart_taux) et
tx_versements(age, ecart_taux) qui donne les taux respectives avec la discreti
zation produite par les nappes.
'''
import pandas as pd

if __name__ == '__main__':

    from stock_pel import stock_pel
    import numpy as np
    import time
    from non_parametric_regression import Kernel, R2
    # ==========================================================================
    # nappe pour le taux de cloture.
    # ==========================================================================

    print('Computing nappe pour taux cloture')

    t0 = time.time()
    X = stock_pel.values[:, 0:2]
    encours = stock_pel.values[:, 2]
    clotures = stock_pel.values[:, 4]
    tx_cloture = clotures / encours

    RS = np.random.RandomState(13062013)
    model_clotures = Kernel(X=X, Y=tx_cloture, h=[1, 2/100], RS=RS)

    nappe_clotures = pd.DataFrame(columns=range(-50, 51),
                                  index=range(180))

    for age in nappe_clotures.index:
        for ecart_taux in nappe_clotures.columns:
            nappe_clotures[ecart_taux][age]\
                 = model_clotures.y_hat(x=[age, ecart_taux / 1000])

    nappe_clotures.to_csv('nappe_clotures.csv')

    t1 = time.time()

    print(f'on a finit avec les clotures en {t1-t0:.2f} secondes \n')

    # ==========================================================================
    # nappe pour le taux de versement.
    # ==========================================================================

    print('Computing nappe de versement')

    stock_pel = stock_pel[stock_pel['age'] < 120]
    X = stock_pel.values[:, 0:2]
    encours = stock_pel.values[:, 2]
    versements = stock_pel.values[:, 3]
    tx_versement = versements / encours

    RS = np.random.RandomState(13062013)
    model_versements = Kernel(X=X, Y=tx_versement, h=[1, 2/100], RS=RS)

    nappe_versements = pd.DataFrame(columns=range(-50, 51),
                                    index=range(120))

    for age in nappe_versements.index:
        for ecart_taux in nappe_versements.columns:
            nappe_versements[ecart_taux][age]\
                = model_versements.y_hat(x=[age, ecart_taux / 1000])

    nappe_versements.to_csv('nappe_versements.csv')

    t2 = time.time()

    print(f'on a finit avec les versements en {t2-t1:.2f} secondes \n')

else:
    nappe_clotures = pd.read_csv('nappe_clotures.csv', sep=',', index_col=0)
    nappe_clotures.columns = range(-50, 51)
    nappe_versements = pd.read_csv('nappe_versements.csv',
                                   sep=',', index_col=0)
    nappe_versements.columns = range(-50, 51)


def tx_clotures(age, ecart_taux):
    if age > 179:
        return 1
    else:
        column = min(max(round(ecart_taux * 1000), -50), 50)
        return nappe_clotures[column][age]


def tx_versements(age, ecart_taux):
    if age > 119:
        return 0
    else:
        column = min(max(round(ecart_taux * 1000), -50), 50)
        return nappe_versements[column][age]

if __name__ == '__main__':
    # on calcule les coefficients de determination pour les test data en
    # utilisant les functions tx_clotures et tx_versements construites a
    # partir des nappes comportamentales (discretitation). Le but c'est
    # d'estimer la perte de performance due a la discretization.

    X_clotures = model_clotures.X_test
    T_clotures = model_clotures.Y_test
    T_hat_clotures = np.array([tx_clotures(*x)
                               for x in list(X_clotures)])

    R2_c = R2(T_clotures, T_hat_clotures)
    print(f'Pour les clotures, le R2 est egale a {R2_c}')

    X_vsmt = model_versements.X_test
    T_vsmt = model_versements.Y_test
    T_hat_vsmt = np.array([tx_versements(*x)
                           for x in list(X_vsmt)])

    R2_v = R2(T_vsmt, T_hat_vsmt)
    print(f'Pour les versements, le R2 est egale a {R2_v}')

# Comentaire: Il n y a pas de perte de performance du modele avec
# la discretization:
# Pour les clotures, le R2 est egale a 0.5889335643004382
# Pour les versements, le R2 est egale a 0.9816484973355254
