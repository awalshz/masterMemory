import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def obtenir_taux(string):
    '''
    string = 'Euribor 3M.txt' ou 'Euribor 1A.txt' pour le taux
    euribor 3 mois ou 1 an respectivement.
    la function return la liste a avec le taux euribor mois par mois
    depuis janvier 1999.
    '''
    euribor = pd.read_csv(string, sep='\t')

    # ====================================================
    # On transforme la matriz euribor_3m a une liste
    vecteur = []
    for annee in map(str, range(1999, 2020)):
        vecteur += list(euribor[annee].values / 100)

    # on elimine les derniers 4 valeurs manquantes
    return vecteur[:-4]

# on obtient le taux euribor 3m et euribor 1 an :
euribor_3m = obtenir_taux('Euribor 3M.txt')
euribor_1a = obtenir_taux('Euribor 1A.txt')


# on obtient le taux OAT 10 ans du site de la banque de france,
# c'est de donnes mensuels, on prende l'observation la plus haut.

OAT = pd.read_csv('OAT 10A.txt', sep=';')
OAT_10a = list(OAT['Plus Haut'].values[::-1] / 100)

# =====================================================================
# On va calculer la matrix de variance covariance de 3 taux precedentes
# =====================================================================

matrice_taux = np.array([  # euribor_3m,
                        euribor_1a,
                        OAT_10a])


def taux_epargne_PEL(mois=None, anne=None, num_mois=None):
    '''
    donne le taux pel de la phase epargne pour un PEL ouvert a la date
    mois/anne. On ne considere que des PEL ouverts apres 1999.
    La variable num_mois est la date en nombre de mois depuis janvier
    1999. (janvier 1999 est num_mois = 0)

    Par example : Pour obtenir le taux PEL de la phase epargne de un PEL
    ouvert en juin 2000, on peut ecrire taux_epargne_PEL(6, 2000) ou
    taux_epargne_PEL(num_mois=17)

    '''
    if num_mois is None:
        num_mois = (anne-1999) * 12 + mois

    if num_mois < 7:
        return 2.9/100
    elif num_mois < 30:
        return 2.61/100
    elif num_mois < 55:
        return 3.27/100
    elif num_mois < 193:
        return 2.5/100
    elif num_mois < 205:
        return 2/100
    elif num_mois < 211:
        return 1.5/100
    else:
        return 1/100


def taux_pret_PEL(mois=None, anne=None, num_mois=None):
    '''
    Pour detailles des variables voir la function taux_epargne_PEL
    Le taux de  pret du PEL est egale a le taux de phase + 1.7% pour
    les PEL ouvert avant fevrier 2015 ou  + 1.2% pour le PEl ouvert
    avant
    '''

    if num_mois is None:
        num_mois = (anne-1999)*12 + mois

    if num_mois <= 193:
        return taux_epargne_PEL(mois=mois, anne=anne, num_mois=num_mois)\
                              + 1.7 / 100
    else:
        return taux_epargne_PEL(mois=mois, anne=anne, num_mois=num_mois)\
                              + 1.2 / 100


# on obtient une liste avec le taux pel de epargne et pret pour les 248
# mois de janvier 1999 a aout 2019.

taux_pel_epargne = []
taux_pel_pret = []
for k in range(248):
    taux_pel_epargne += [taux_epargne_PEL(num_mois=k+1)]
    taux_pel_pret += [taux_pret_PEL(num_mois=k+1)]


# on regarde graphiquement la relation entre le taux pel (phase epargne)
# et le taux Euribor 1 an.
if __name__ == '__main__':
    print(f'matrice_taux est de dimension {matrice_taux.shape} et \n\
          sa matrice de variance covariance est : \n')

    print(np.cov(matrice_taux))

    t = np.arange(248)
    # plt.plot(t, np.array(euribor_1a), '--', label="euribor",  t,  # 'bs'
    #          np.array(taux_pel_epargne), '.-', label="taux pel")  # 'g^'
    plt.plot(t, np.array(euribor_1a), color="blue",
             linewidth=1.5, linestyle="--", label="euribor")
    plt.plot(t, np.array(taux_pel_epargne), color="red",
             linewidth=1.5, linestyle="-.", label="taux pel")
    plt.legend(loc='upper right')
    plt.show()

    # =========================================================================
    # soit x le taux euribor  1 an et y le taux pel (phase epargne). Pour
    # choisir le taux pel, on considere le taux euribor 1 an comme
    # un proxi du marche. Il y a une relation entre le taux pel et le taux
    # euribor : y = a + bx + e. On calcule les coeficients a et b par moindre
    # carres :

    x = np.array(euribor_1a)
    y = np.array(taux_pel_epargne)
    matriz_var = np.cov(x, y)
    b = matriz_var[0, 1] / matriz_var[0, 0]
    a = y.mean() - b * x.mean()
    print([a, b])
    coef_correlation = b * x.std() / y.std()
    print(coef_correlation)

    # donc a = 0.02 et b = 0.25. Le coefficient de correlation entre le taux
    #  euribor et le taux pel est 0.66.

    # Commentaire: pour le cloture et les versements exceptionels, le gens
    # prend en consideration le taux pel par rappor au conditions du marche.
    # Le proxi du marche a utiliser est, du au regression lineaire decrit
    # precedement :
    # r = 0.02 +0.25*taux_euribor_1a. Si taux_pel-r est eleve le gens realisent
    # des versement exceptionnels et peut de clotures. Au contraire, si
    # taux_pel-r est basse (le conditions du marche sont plus rentables que les
    # taux pel du contrat... qui va rester fixe), le gens vont cloturer le pel
    # et realiser peu des versements exceptionnels.
