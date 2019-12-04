import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
    from client import Client
    import get_data as gd


def generator_strate(mois_origine, num_contrats=1000):
    '''
    on genere 1000 contrats pel ouvert en mois_origine.
    On donne les encours, versements, clotures, age et
    ecart de taux de la population total, c'est a dire
    on va agreger toutes les clients et les representer dans
    un seule strate.
    '''

    RS = np.random.RandomState(198304+mois_origine)
    dernier_mois = min(mois_origine + 179, 247)

    # pour chaque client ou contrant, la taille du vecteur
    # des encours est size avec
    num_encours = dernier_mois - mois_origine + 1

    data = np.zeros([num_encours, 3])  # la premier colonne
    # aura l'age du pel, la deuxieme, l'ecart de taux euribor 3m
    # avec taux du pel, la quatrime les encours, la cinquieme
    # les versements et la sixieme les clotures.

    for n in range(num_contrats):
        montant_souhaite = 36 * 1000
        duree_souhaite = 7.5 * 12
        client = Client(mois_origine, montant_souhaite, duree_souhaite)
        client.simulation_encours(RS)
        data[:, 0] += client.encours
        data[:, 1] += client.versements
        data[:, 2] += client.clotures

    ages = range(num_encours)
    ecart_taux = np.array(gd.euribor_3m[mois_origine:mois_origine +
                          num_encours]) - client.tx_epargne

    data = np.column_stack((range(num_encours), ecart_taux, data))

    return data

if __name__ == '__main__':
    time0 = time.time()

    columns_names = ['age',
                     'ecart_taux',
                     'encours',
                     'versements',
                     'clotures',
                     ]

    stock_pel = pd.DataFrame(columns=columns_names)

    for strate in range(248):
        print(f'generation de la strate numero {strate}')
        data_strate = pd.DataFrame(generator_strate(strate),
                                   columns=columns_names)
        stock_pel = stock_pel.append(data_strate, ignore_index=True)

    stock_pel.to_csv('stock_pel.csv')

    time1 = time.time()

    print(f'on a genere le fichier stock_pel.csv en {time1-time0:2f} s \n')

else:
    stock_pel = pd.read_csv('stock_pel.csv', sep=',', index_col=0)
