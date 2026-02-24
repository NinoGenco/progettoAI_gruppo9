import numpy as np
from KNNAlgorithm.CalculateDistance.Strategy.DistanceStrategy import DistanceStrategy

class EuclideanDistance(DistanceStrategy):

    """ Questa classe implementa la strategia di calcolo della distanza euclidea. Per ottimizzare le prestazioni abbiamo
    omesso la radice quadrata, in quanto ci interessa solo ordinare le distanze tra i punti, senza bisogno dei valori esatti."""

    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:

        """Questa funzione calcola la somma dei quadrati delle differenze tra i componenti corrispondenti dei due vettori.
        Inoltre include un controllo preventivo sulle loro dimensioni, in modo da garantire coerenza matematica.

        Parametri:
                  x1 -> Primo vettore numerico (Solitamente un campione estratto dal Training Set).
                  x2 -> Secondo vettore numerico (Solitamente un singolo campione estratto dal Test Set da classificare).

        Risultati -> Valore numerico float che rappresenta la distanza calcolata tra x1 e x2."""

        # Verifico che i vettori abbiano la stessa dimensione.
        if x1.shape != x2.shape:
            # Se la condizione if non è verificata blocco l'esecuzione del programma stampando un messaggio di errore.
            raise ValueError("I vettori sono incompatibili perchè hanno dimensioni diverse.")

        # Eseguo il calcolo della distanze, restituendo il risultato finale.
        return np.sum(pow(x1-x2, 2))