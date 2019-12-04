import numpy as np
from comportemental.get_data import taux_epargne_PEL, euribor_3m
from simulation_taux.gpp import simu_euribor3m, simu_oat10a
from comportemental.nappe import tx_clotures, tx_versements
from comportemental.simulation_erreurs import erreurs_simule
import matplotlib.pyplot as plt

[num_sim, num_mois] = simu_euribor3m.shape


class Pel:
    def __init__(self, age, encours0=1, erreurs=False):
        '''
        class qui contient a generation de PEL, age, c'est
        l'age du PEL ajourhui, et encours le montant du PEL.
        L'age doit etre inferieur a 178 car un pel de 179 mois
        se ferme le mois prochain, donc il faut pas le prendre
        en compte pour la provision.
        '''

        self.age = age
        self.encours0 = encours0
        mois_pel = 247 - age  # mois d'origine de pel, aout 2019
        # est le mois 247. Janvier 1999 le mois 0.
        self.taux_pel = taux_epargne_PEL(num_mois=mois_pel)
        self.encours = self.get_encours(erreurs)

    def taux_clotures(self, erreurs=False):
        tx_cl = []
        for simu in range(num_sim):
            tx_cl_i = []
            for mois in range(num_mois):
                if erreurs:
                    E = erreurs_simule[simu, mois]
                else:
                    E = 0

                tx = tx_clotures(self.age + mois,
                                 simu_euribor3m[simu, mois] -
                                 self.taux_pel) + E

                tx_cl_i += [tx]

            tx_cl += [tx_cl_i]

        return np.array(tx_cl)

    def taux_versements(self):
        tx_vs = []
        for simu in simu_euribor3m:
            tx_vs_i = [tx_versements(self.age + i, simu[i] - self.taux_pel)
                       for i in range(num_mois)]
            tx_vs += [tx_vs_i]
        return np.array(tx_vs)

    def get_encours(self, Erreurs=False):
        encours = self.encours0 * np.ones([num_sim, 1])
        versements = self.taux_versements()
        clotures = self.taux_clotures(Erreurs)

        for mois in range(num_mois - 1):
            encours = np.append(encours,
                                (encours[:, -1] +
                                 encours[:, -1] * (versements[:, mois] -
                                                   clotures[:, mois])
                                 ).reshape(num_sim, 1),
                                axis=1)

        return encours

    def provision(self, q=0):
        '''
        q es le percentile
        '''
        return (DF * (self.encours - np.percentile(self.encours,
                                                   q=0, axis=0)) *
                (1 / 12) * (self.taux_pel - taux_ref)).sum(axis=1)


if __name__ == '__main__':
    # ==============================================================
    # on va faire un graphique de une surface en risque :
    #  encours probable - encours certain
    pel = Pel(20, erreurs=True)
    t = np.arange(180)
    plt.plot(t, np.percentile(pel.encours, q=97.5, axis=0), color="blue",
             linewidth=1.5, linestyle="-", label="encours probable")
    plt.plot(t, np.percentile(pel.encours, q=2.5, axis=0), color="red",
             linewidth=1.5, linestyle="--", label="encours certain")
    plt.legend(loc='upper right')
    plt.show()

# ==============================================================
# taux d'acutalisation : du CNC, euribor 3m moyennes sur 12 mois

if __name__ == '__main__':
    euribor = euribor_3m[-13:-2] * np.ones([num_sim, 11])
    euribor = np.append(euribor, simu_euribor3m, axis=1)
    for i in range(num_mois):
        euribor[:, -i-1] = euribor[:, -i-13:-i-1].mean(axis=1)

    euribor_3m_moy = euribor[:, 11:]

    # Les factor d'actualisation d'un mois au mois i est egale a :

    Pij = (1 / (1 + euribor_3m_moy)) ** (1 / 12)

    # Les factors d'actualisation s'obtient de la formule
    # P0j = P0,1 * P1,2 * P2,3 * ... * Pj-1,j

    DF = np.ones([num_sim, num_mois])
    for i in range(1, num_mois):
        DF[:, i] = DF[:, i-1] * Pij[:, i-1]

    np.savetxt('Discount Factor', DF)

    # on fait un graphique de deux simulations de factor de discount.
    t = np.arange(180)
    plt.plot(t, DF[0], color="blue",
             linewidth=1.5, linestyle="-", label="DF pour siulation 0")
    plt.plot(t, DF[1], color="red",
             linewidth=1.5, linestyle="--", label="DF pour siulation 1")
    plt.legend(loc='upper right')
    plt.show()

else:
    DF = np.loadtxt('Discount Factor')

# ==================================================================
# taux reference : On prend le taux OAT 10 ans plafonée a 0.75% (plafond
# du livret A historique. C'est le gouvernement qui decide le taux)
# Son taux est déterminé par une formule liée aux taux courts et à l’inflation,
#  mais la Banque de France peut proposer au gouvernement de déroger
# exceptionnellement à la règle. Ce taux sert de référence pour déterminer
# les taux de plusieurs autres supports d’épargne réglementée.
# Depuis août 2015, le taux du Livret A est de 0,75%.

taux_ref = np.maximum(simu_oat10a, 0.75 / 100)

if __name__ == '__main__':
    # on fait un graphique de deux simulations de factor de discount.
    t = np.arange(180)
    plt.plot(t, taux_ref[0], color="blue",
             linewidth=1.5, linestyle="-", label="Taux Ref pour siulation 0")
    plt.plot(t, taux_ref[1], color="red",
             linewidth=1.5, linestyle="--", label="Taux Ref pour siulation 1")
    plt.legend(loc='upper right')
    plt.show()
