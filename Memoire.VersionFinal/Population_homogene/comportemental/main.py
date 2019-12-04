from stock_pel import stock_pel
import pandas as pd
import numpy as np
import time
from non_parametric_regression import Kernel

X = stock_pel.values[:, 0:2]
encours = stock_pel.values[:, 2]
clotures = stock_pel.values[:, 4]
tx_clotures = clotures / encours

RS = np.random.RandomState(13062013)

model = Kernel(X=X, Y=tx_clotures, RS=RS)

print(f'l ecart type de la variable tx_clotures est {model.Y_test.std()}')

stock_pel = stock_pel[stock_pel['age'] < 120]
X = stock_pel.values[:, 0:2]
encours = stock_pel.values[:, 2]
versements = stock_pel.values[:, 3]
tx_versement = versements / encours

RS = np.random.RandomState(13062013)

model = Kernel(X=X, Y=tx_versement, RS=RS)

print(f'l ecart type de la variable tx_versements est {model.Y_test.std()}')
