import numpy as np
from KNNAlgorithm.CalculateDistance.Strategy.DistanceStrategy import DistanceStrategy

class EuclideanDistance(DistanceStrategy):

    """ Questa classe concreta implementa la strategia di calcolo della distanza euclidea. Per ottimizzare le
    prestazioni omettiamo la radice quadrata, in quanto ci interessa solo ordinare le distanze tra i punti,
    senza trovare i valori esatti."""

    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:

        """ Questa funzione calcola la somma dei quadrati delle differenze tra i componenti corrispondenti dei due
        vettori. Inoltre include un controllo preventivo sulle dimensioni di tali vettori, in modo da garantire
        coerenza matematica.

        Parametri: x1: Primo vettore numerico (Solitamente un campione del Training Set).
                   x2: Secondo vettore numerico (Solitamente il punto di Test Set da classificare).

        Risultati: Valore numerico float che rappresenta la distanza calcolata tra x1 e x2."""

        # Verifico che i vettori abbiano la stessa dimensione per evitare broadcasting errati.
        if x1.shape != x2.shape:
            raise ValueError("I vettori sono incompatibili perchè hanno dimensioni diverse.")

        return np.sum(pow(x1-x2, 2))