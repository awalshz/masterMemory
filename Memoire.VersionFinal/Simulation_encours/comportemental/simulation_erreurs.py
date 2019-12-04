'''
Ici on va simuler une matrize 1000 * 180 des erreurs
de taux de cloture de la regression non parametrique
'''

import numpy as np
if __name__ == '__main__':
    from erreurs import E_cloture

    T = E_cloture.size

    RS = np.random.RandomState(204)
    erreurs_simule = []
    for n in range(1000):
        erreurs_simule += [E_cloture[RS.randint(0, T, size=180)]]

    erreurs_simule = np.array(erreurs_simule)
    np.savetxt('erreurs_simule', erreurs_simule)
else:
    erreurs_simule = np.loadtxt('erreurs_simule')
