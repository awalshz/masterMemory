'''
ici on va comparer l'encours minimun d un pel avec et sans les
erreurs
'''
import numpy as np
from pel import Pel
import matplotlib.pyplot as plt

pels = Pel(20, erreurs=False)
pele = Pel(20, erreurs=True)
t = np.arange(180)
plt.plot(t, pels.encours.min(axis=0), color="blue",
         linewidth=1.5, linestyle="-",
         label="encours certain sans residus")
plt.plot(t, np.percentile(pele.encours, q=2.5, axis=0), color="red",
         linewidth=1.5, linestyle="--",
         label="encours certain avec residus")
plt.legend(loc='upper right')
plt.show()
