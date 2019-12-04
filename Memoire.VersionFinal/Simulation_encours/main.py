import pandas as pd
import numpy as np
from simulation_taux.gpp import mcov_theorique
from comportemental.get_data import matrice_taux
from scipy.optimize import minimize

# ===================================================================
# on va trouver les coefficients a, b, sigma, eta et rho en essayant
# de approximer la matriz de variance convariance donnes par les taux
# euribor 3 mois, euribor 1 an et oat 10 ans.
dif_taux = matrice_taux[:, -120:-1] - matrice_taux[:, -121:-2]
mcov_estime = np.cov(dif_taux)
# on va eliminer un valeur a de la matrice, la variance de euribor 1a :
# parce que on a 6 valeurs a trouver et 5 parametres
# mcov_estime[0, 1] = 1
# mcov_estime[1, 0] = 1

T = [1, 5]  # les 3 maturites des taux observes.


parametres = {'a': 0.4, 'b': 0.9, 'sigma': 0.005,
              'eta': 0.003, 'rho': -0.7}
minimun = 10 ** 5
for essai_num in range(1000000):
    a = np.random.random()
    b = np.random.random()
    sigma = 0.005  # np.random.random()
    eta = np.random.random()
    rho = -0.7  # -0.5 - 0.3 * np.random.random()
    cov_t = mcov_theorique(T, a, b, sigma, eta, rho)
    # cov_t[0, 1] = 1
    # cov_t[1, 0] = 1
    distance = ((cov_t / mcov_estime - np.ones([2, 2])) ** 2).sum()
    if distance < minimun:
        parametres = {'a': a, 'b': b, 'sigma': sigma,
                      'eta': eta, 'rho': rho}
        minimun = distance
'''
def f(x):
    cov_t = mcov_theorique(T, *x)
    distance = (((cov_t / mcov_estime - np.ones([3, 3])) * pesos) ** 2).sum()
    return distance

g = minimize(f, [0.4, 0.9, 0.03, 0.03, -1])
'''

print(f'Voici les parametres: \n {parametres} \n')

print(f'La matriz de convariance avec les parametres precedentes\n \
      est en termes de porcentages de la matriz observe : \n')
cov_t = mcov_theorique(T, *parametres.values())
print(cov_t / mcov_estime)

print(cov_t)

print(mcov_estime)
