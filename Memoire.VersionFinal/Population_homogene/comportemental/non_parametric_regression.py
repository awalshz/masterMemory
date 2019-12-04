'''
module que calcule la regression non parametrique o kernel regression
'''

import numpy as np
from sklearn.model_selection import train_test_split


class Kernel:
    def __init__(self, X, Y, h=None, test_size=1/3, RS=None):
        '''
        X est une matriz (numpy 2D array) avec les variables explicatives

        Y est la variable independante.

        h est la fenetre de lissage pour le vecteur des observations.

        f est la function kernel a utiliser : f = gaussienne par default.

        RS le numpy random state pour avoir la seed
        '''
        if test_size == 0:
            self.X_train = X
            self.X_test = None
            self.Y_train = Y
            self.Y_test = None
        else:
            (self.X_train,
             self.X_test,
             self.Y_train,
             self.Y_test) = train_test_split(X,
                                             Y,
                                             test_size=test_size,
                                             random_state=RS)
        self.h = h

    def y_hat(self, x, h=None):
        '''
        donne le valeur estime du y associe a une observation x
        de la regression non parametrique avec function kernel
        f (gaussien par default) et h la fenetre de lissage fenetre.
        x et h sont de largeur le nombre de colonnes des observatios X.
        La regression non parametrique utilise uniquement les
        observations du train data.
        '''

        num_obs, num_var = self.X_train.shape
        if h is None:
            h = self.h

        product = np.ones(num_obs)
        for i in range(num_var):
            Z_i = np.exp(-((self.X_train[:, i] - x[i]) / h[i])**2)
            product = product*Z_i

        return np.inner(self.Y_train, product)/product.sum()

    def Y_hat(self, X, h=None):

        Y_hat = []

        for x in list(X):
            Y_hat += [self.y_hat(x=x, h=h)]

        return np.array(Y_hat)

    def R2(self, X=None, Y=None, h=None):
        '''
        Return le coefficient de determination R2, base sur la
        performance du model sur les observations X, Y (test data
        par default).
        '''
        if X is None or Y is None:
            X = self.X_test
            Y = self.Y_test

        if Y is not None:
            Y_hat = self.Y_hat(X=X, h=h)
            return(R2(Y, Y_hat))
        else:
            return(None)


def R2(Y, Y_hat):
    return 1 - (1 / Y.size) * ((Y - Y_hat) ** 2).sum() / Y.var()
