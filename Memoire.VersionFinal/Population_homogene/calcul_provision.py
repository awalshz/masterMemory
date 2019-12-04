import numpy as np
import time
from comportemental.stock_pel import stock_pel
from pel import Pel
import matplotlib.pyplot as plt


# on considere la base des pel a la date d'ajourhui
stock_pel = stock_pel.iloc[[-1 - n * (n + 1) / 2 for n in range(179)]]
# le dernier gener la liste de pel de age 0 au age 178  a la date
# d'ajourdhui

if __name__ == '__main__':

    loi_provision = []
    for age in range(179):
        print(f'computing provision of strate {age}')
        time0 = time.time()
        strate = Pel(age=age, encours0=stock_pel['encours'].iloc[age])
        loi_provision += [strate.provision()]
        time1 = time.time()
        print(f'Done in {time1 - time0:2f} seconds \n')

    np.savetxt('loi_provision', np.array(loi_provision))

else:
    loi_provision = np.loadtxt('loi_provision')

# les pel ouverts a partir de setembre 2004 apartient a la 8,9
# 10, 11 et 12 generation. On va calculer la provision de chaque generation.

valeurs = np.unique(stock_pel['ecart_taux'].values)
generation = []
encours = []
for g in range(4):  # la generation 8 sera dans le premier element de la liste
    generation_g = stock_pel['age'][stock_pel['ecart_taux'] ==
                                    valeurs[g]].values
    generation_g = list(map(int, generation_g))  # on obtien la liste avec les
    # ages de la generation 8
    generation += [generation_g]
    encours_g = stock_pel['encours'][stock_pel['ecart_taux'] ==
                                     valeurs[g]].values.sum()
    encours += [encours_g]

prov_generation = []  # liste avec la  loi provisions par generation,
for g in range(4):
    prov_g = loi_provision[generation[g]].sum(axis=0)  # on fait la
    # somme agregé des pel qui appartient a la generation.
    prov_g = np.maximum(prov_g, 0)  # la provision ne peut pas etre negative.
    prov_generation += [prov_g]

prov_generation = np.array(prov_generation)

provision_total = prov_generation.sum(axis=0)


montant_total = stock_pel['encours'].sum()
pourcentage = provision_total.mean() / montant_total

n, bins, patches = plt.hist(provision_total * 1e-6, 20, label=None,
                            density=True, facecolor='g', alpha=0.75)
plt.xlabel('Provision en M€')
plt.ylabel('Probability')
plt.title('Loi empirique')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# les percentiles a 10, 25, 50, 75, 90, 95 %, mean et sigma:
statistiques_keys = ['q10', 'q25', 'q50', 'q75', 'q90',
                     'q95', 'mean', 'sigma']
q = [10, 25, 50, 75, 90, 95]
statistiques = [np.percentile(provision_total, x) / 1e+6 for x in q]
statistiques += [provision_total.mean() / 1e+6]
statistiques += [provision_total.std() / 1e+6]
dict_statis = dict(zip(statistiques_keys, statistiques))
