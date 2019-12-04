import get_data as gd
import numpy as np
import time


class Client:
    def __init__(self, mois_origine, montant_souhaite,
                 duree_souhaite):
        '''
        mois_origine c'est le mois de ouverture du pel... le mois i
        correspond a i mois apres janvier 1999. Le mois 0 est janvier 1999.

        montant_souhaite est le montant a epargner prevue :
        c'est un valeur entre 10000 et 61200 (le plafond du PEL)

        duree_souhaite = la duree prevue par le client de la phase epargne
        logement.
        '''

        self.tx_epargne = gd.taux_epargne_PEL(num_mois=mois_origine)
        self.tx_pret = gd.taux_pret_PEL(num_mois=mois_origine)
        self.mois_origine = mois_origine

        # le rendement mensuel de la phase epargne est donc
        r = (1 + self.tx_epargne) ** (1 / 12) - 1

        self.tx_epargne_mensuel = r

        # la mensualite pour epargne le montant suhaite dans la duree souhaite
        self.montant_souhaite = montant_souhaite
        self.mensualite = r * montant_souhaite / ((1+r) ** duree_souhaite - 1)

    def simulation_encours(self, RS):
        '''
        funcion pour simuler les encours, versement et clotures
        du pel, en funcion du conditions du marche du contrat
        pel. RS est un numpy random state pour le seed.
        '''

        if RS is None:
            RS = np.random.RandomState()

        r = self.tx_epargne_mensuel

        encours = [self.mensualite]  # c est l'encours a la date d'ouverture

        versements = [self.mensualite]  # le premier mois il n'a pas de
        # versement exceptionnel.

        clotures = [0]  # c'est l'ouverture du PEL, donc il n y a pas de
        # cloture

        dernier_mois = min(self.mois_origine + 179, 247)  # c'est le dernier
        # mois a simuler :  soit le pel arrive a sa age maximal 180 mois,
        # soit le pel arrive a le mois actuel; le mois 248 = aout 2019.

        for mois in range(self.mois_origine + 1, dernier_mois + 1):
            if encours[-1] == 0:
                # si le dernier montant des encours est 0, le PEL est fermee,
                # il n y a pas donc des versements ni clotures pour les mois
                # suivantes.
                encours += [0]
                versements += [0]
                clotures += [0]
            else:
                # On calcule demande  si le clien veut cloturer le pel.
                # Le taux cloture annuel est definie comme p + ref, avec
                # p une variable qui depend de l'age du pel et ref une variable
                # qui depend des taux du pel et des conditions de marche.
                # On prend ref = 5*(taux marche -taux pel). Plus grande ref,
                # plus grand le desir de cloturer le PEL.
                # Par example: Si taux_marche - taux_pel est de 10%
                # (un grand ecart) le taux de cloture annuel va aumenter
                # 50 pbs.
                # Le taux de marche a comparer avec le taux est :
                # 0.02 + 0.25 * euribor_1a (la regression linaire fait dans
                # get_data.py)

                taux_marche = 0.02 + 0.25 * gd.euribor_1a[mois]
                ref = 5 * (taux_marche - self.tx_epargne)

                age_pel = mois - self.mois_origine
                if age_pel < 48:
                    cloturer_pel = veut_cloturer(age_pel=age_pel,
                                                 taux_pel=self.tx_epargne,
                                                 ref=ref,
                                                 RS=RS
                                                 )
                else:
                    cloturer_pel = veut_cloturer(
                        age_pel=age_pel,
                        taux_pel=self.tx_epargne,
                        ref=ref,
                        RS=RS,
                        montant_souhaite=self.montant_souhaite,
                        encours_2=encours[-2],
                        encours_1=encours[-1]
                        )

                if cloturer_pel:
                    clotures += [min(encours[-1]*(1+r), 61200)]
                    versements += [0]
                    encours += [0]
                else:
                    if round(encours[-1] * (1+r), 4) >= 61200:
                        # l'encours du pel est plafonee a 61200€.
                        encours += [61200]
                        versements += [0]
                        clotures += [0]
                    else:
                        # le montant a verser va incrementer si le taux
                        # marche es base par rapport au taux du pel, c'est
                        # a dire si -ref est eleve.
                        if age_pel < 120:
                            montant_verser = self.mensualite\
                                        * max(1, 1-ref+0.25 * RS.normal())
                        else:
                            montant_verser = 0

                        versements += [min(61200 - encours[-1] * (1+r),
                                           montant_verser)]
                        clotures += [0]
                        encours += [encours[-1] * (1 + r) + versements[-1]]

        self.encours = np.array(encours)
        self.versements = np.array(versements)
        self.clotures = np.array(clotures)


def veut_cloturer(age_pel,
                  taux_pel,
                  ref,
                  RS=None,
                  montant_souhaite=None,
                  encours_2=None,
                  encours_1=None,
                  taux_marche=None):

    '''
    resulta de la funcion = True si le client cloture le pel,
    = False si ne le souhaite pas. La variable encours_2 est
    l'encours du pel 2 mois avant et encours_1 celui du mois
    precedent. Les 2 variables servent a determiner si le
    client vient de attend son objectif = montant souhaite.
    RS= numpy random state pour le seed.
    La variable ref (reference) est la variable de adjustment.
    Il change le taux de cloture en function du marche et du
    taux pel. Le taux cloture annuel est  p + ref.
    '''

    if age_pel < 24:
        p = 1 - (max(0, 0.975 - ref)) ** (1 / 12)  # le taux cloture
        # est de 2.5% annuel le deux premiers annes de ouverture du pel
    elif age_pel < 48:
        p = 1 - (max(0, 0.95 - ref)) ** (1 / 12)  # 5% annuel avant le 4 anne.
    else:
        if round(encours_1, 4) < montant_souhaite:
            # le client n'a pas encore arriver a epargner le
            # montant souhaite, le taux cloture est de 10%
            # annuel.
            p = 1 - (max(0, 0.9 - ref)) ** (1 / 12)
        elif round(encours_2, 4) < montant_souhaite:
            # le client vient d'arriver a epargner son montant
            # souhaite. Sont taux cloture est de 60 %
            p = min(1, 0.6 + ref)
        else:
            # si le client ne cloture pas le pel quand arrive
            # a epargner le montant souhaite, alors il va le
            # cloture aux taux de 30% annuel
            p = 1 - (max(0, 0.7 - ref)) ** (1/12)

    if RS is None:
        RS = np.random.RandomState()

    return RS.uniform() < p
