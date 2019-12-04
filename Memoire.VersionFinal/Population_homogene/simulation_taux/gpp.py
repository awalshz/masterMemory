import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def B(z, t):
    return (1 - np.exp(-z * t)) / z


def V(t, a, b, sigma, eta, rho):
    S1 = ((sigma ** 2) / (a ** 2)) * (t - 2 * B(a, t) + B(2 * a, t))
    S2 = ((eta ** 2) / (b ** 2)) * (t - 2 * B(b, t) + B(2 * b, t))
    S3 = (2 * rho * sigma * eta / (a * b)) * \
         (t - B(a, t) - B(b, t) + B(a + b, t))
    return S1 + S2 + S3


def cov_theorique(T1, T2, a, b, sigma, eta, rho):
    S1 = (sigma ** 2) * B(a, 1 / 12) * B(a, T1) * B(a, T2) / (T1 * T2)
    S2 = (eta ** 2) * B(b, 1 / 12) * B(b, T1) * B(b, T2) / (T1 * T2)
    S3 = rho * eta * sigma / (a + b) *\
        (a * B(a, 1 / 12) + b * B(b, 1 / 12)) *\
        (B(a, T1) * B(b, T2) + B(b, T1) * B(a, T2)) / (T1 * T2)

    return S1 + S2 + S3


def mcov_theorique(T, a, b, sigma, eta, rho):
    '''
    calcul de la matriz de variance covariance theorique.
    T est une liste avec les maturites
    '''
    n = len(T)
    matrice = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            T1 = T[i]
            T2 = T[j]
            matrice[i, j] = cov_theorique(T1, T2, a, b, sigma, eta, rho)

    return matrice


class G2pp:
    '''
    class qui simule une trajectoire (x_t, y_t) du modele G2++
    '''

    def __init__(self, a, b, sigma, eta, rho):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.eta = eta
        self.rho = rho
        self.X = np.array([])
        self.Y = np.array([])

    def simula(self, x0=0, y0=0, pas=1/12, num_pas=12, RS=None):
        '''
        pas le pas de la simulation, on simule par mois
        par defaut.

        num_pas= nombre de variable a simuler dans la trajectoire.

        RS = numpy random state pour tracker la seed.
        '''
        if RS is None:
            RS = np.random.RandomState()

        Z = [[x0, y0]]
        for t in range(num_pas - 1):
            x = Z[-1][0]
            y = Z[-1][1]

            m_cov = [
                     [self.sigma ** 2 * B(2 * self.a, pas),
                      self.rho * self.sigma * self.eta *
                      B(self.a + self.b, pas)],
                     [self.rho * self.sigma * self.eta *
                      B(self.a + self.b, pas),
                      self.eta ** 2 * B(2 * self.b, pas)]
                ]

            m_cov = np.array(m_cov)

            N = RS.multivariate_normal(mean=np.array([0, 0]), cov=m_cov)

            x1 = x*np.exp(-self.a*pas) + N[0]

            y1 = y*np.exp(-self.b*pas) + N[1]

            Z += [[x1, y1]]

        Z = np.array(Z)
        return Z

    def simula_XY(self, x0=0, y0=0, pas=1/12, num_pas=12, num_sim=1, RS=None):
        X = []
        Y = []
        for i in range(num_sim):
            Z = self.simula(x0, y0, pas, num_pas, RS)
            X += [Z[:, 0]]
            Y += [Z[:, 1]]
        return (np.array(X), np.array(Y))  # X.shape = [num_sim, num_pas]

    def add_XY(self, x0=0, y0=0, pas=1/12, num_pas=12, num_sim=1, RS=None):
        (X, Y) = self.simula_XY(x0, y0, pas, num_pas, num_sim, RS)
        self.X = X
        self.Y = Y
        self.pas = pas

    def taux(self, maturite, courbe, pas=1/12, num_pas=12,
             num_sim=None, RS=None):
        '''
        simule le trajectoires du taux avec courbe initial f^M=courbe
        et a maturite T. Courbe = f est une fonction t --> f^M(0,t)
        '''

        if num_sim is None:
            X = self.X
            Y = self.Y
            pas = self.pas
            num_sim = X.shape[0]
            num_pas = X.shape[1]

        else:
            (X, Y) = self.simula_XY(x0=0, y0=0, pas=pas,
                                    num_pas=num_pas,
                                    num_sim=num_sim, RS=None)

        T = maturite
        f = np.array([(T + n * pas) * courbe(T + n * pas) -
                     n * pas * courbe(n * pas)
                     for n in range(num_pas)])
        para = [self.a, self.b, self.sigma, self.eta, self.rho]
        Vt = -0.5 * V(T, *para) + 0.5 * np.array([V(T + n * pas, *para) -
                                                 V(n * pas, *para)
                                                 for n in range(num_pas)])

        return (f + Vt + B(self.a, T) * X + B(self.b, T) * Y) / T

if __name__ == '__main__':

    # les valeurs del example de la section 4.2.7 de [BM]
    a, b, sigma, eta, rho = 0.77, 0.082, 0.02, 0.01, -0.7

    # on va simuler 7 trajectoires de taux OAT 3 mois avec la
    # courbe inital OAT de juillet 2019.

    mois = np.array([0, 1, 3, 6, 9, 12, 24, 60, 120, 360]) / 12
    OAT = np.array([-0.4102903226,
                    -0.4102903226,
                    -0.409483871,
                    -0.4404516129,
                    -0.4866451613,
                    -0.4727096774,
                    -0.5133225806,
                    -0.4129354839,
                    -0.0512032258,
                    0.6592903226])

    f_OATi = interp1d(mois, OAT / 100, kind='cubic')

    def f_OAT(t):
        return float(f_OATi(min(t, 30)))

    # on fait un graphique de la courbe de taux avec kernel regression
    m = np.array(range(360)) / 12
    taux = list(map(f_OAT, m))
    plt.plot(m, taux, label="courbe OAT juillet 2019")
    plt.legend(loc='upper left')
    plt.show()

    g2pp = G2pp(a=a, b=b, sigma=sigma, eta=eta, rho=rho)

    RS = np.random.RandomState(19032015)
    g2pp.add_XY(num_pas=120, num_sim=10, RS=RS)
    R1 = g2pp.taux(maturite=3 / 12, courbe=f_OAT)
    R2 = g2pp.taux(maturite=1, courbe=f_OAT)

    cv = cov_theorique(3 / 12, 1, a, b, sigma, eta, rho)

    estimateur = []
    for i in range(10):
        estimateur += [((R1[i, 1:]-R1[i, :-1]) *
                       (R2[i, 1:]-R2[i, :-1])).mean() / cv]

    print(estimateur)

    print(cv)

    # ==============================================================
    # Pour le calcul de la provision, on va generer 1000 simulations
    # de 180 mois de observations de taux euribor 3 mois et euribor
    # 10 ans. Le donnes pour le taux 1 semaine, 1, 3, 6 et 12 mois
    # juillet 2019 sont pris du site
    # https://fr.global-rates.com/taux-de-interets/euribor/2019.aspx
    # toutes les taux sont negative on ajoute a taux a 30 ans a 0%
    # ==============================================================

    # courbe inital euribor de juillet 2019.

    mois = np.array([0, 12/52, 1, 3, 6, 12, 24]) / 12
    euribor = np.array([-0.402,
                        -0.402,
                        -0.395,
                        -0.365,
                        -0.347,
                        -0.283,
                        0])

    f_euribori = interp1d(mois, euribor / 100, kind='cubic')

    def f_euribor(t):
        return float(f_euribori(min(t, 2)))

    m = np.array(range(14)) / 12
    taux = list(map(f_euribor, m))
    plt.plot(m, taux, label="courbe euribor juillet 2019")
    plt.legend(loc='upper left')
    plt.show()

    RS = np.random.RandomState(19032015)
    g2pp.add_XY(num_pas=180, num_sim=1000, RS=RS)
    simu_euribor3m = g2pp.taux(maturite=3 / 12, courbe=f_euribor)
    simu_oat10a = g2pp.taux(maturite=10, courbe=f_OAT)

    np.savetxt('simulation_OAT', simu_oat10a)
    np.savetxt('simulation_Euribor', simu_euribor3m)

else:
    simu_euribor3m = np.loadtxt('simulation_Euribor')
    simu_oat10a = np.loadtxt('simulation_OAT')
