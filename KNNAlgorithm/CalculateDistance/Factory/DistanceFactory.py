from abc import ABC
from KNNAlgorithm.CalculateDistance.Strategy.EuclidianDistance import EuclidianDistance

class DistanceFactory(ABC):

    """ Factory Class per la gestione delle strategie di calcolo della distanza.

    Utilizza il Design Pattern 'Factory Method' per centralizzare la creazione
    delle metriche, garantendo flessibilità e disaccoppiamento tra il core
    dell'algoritmo e le implementazioni matematiche specifiche."""

    @staticmethod
    def get_distance_metric(metric_name: str):

        """ Restituisce un'istanza della strategia di distanza richiesta.

        Il metodo normalizza l'input per evitare errori di case-sensitivity.
        Nota: La logica di calcolo è delegata alla strategia istanziata per
        ottimizzare le performance ed evitare istanziazioni multiple nei cicli.

        :param metric_name: Nome della metrica (es. 'euclidean')
        :return: Istanza di una sottoclasse di DistanceStrategy
        :raises ValueError: Se la metrica richiesta non è supportata"""

        # Normalizzazione dell'input: rimuove spazi e converte in minuscolo
        clean_name = metric_name.lower().strip()

        if clean_name == 'euclidian' or clean_name == 'euclidean':
            return EuclidianDistance()
        else:
            raise ValueError(f"La metrica '{metric_name}' non è supportata o non esiste.")

    """ La funzione di calcolo non è inclusa nella Factory per rispettare il principio 
    di singola responsabilità. Istanziare la strategia una sola volta nel 
    costruttore del KNN, anziché ripetutamente all'interno dei cicli di predizione, 
    previene il degrado delle prestazioni dovuto all'overhead di creazione degli oggetti."""