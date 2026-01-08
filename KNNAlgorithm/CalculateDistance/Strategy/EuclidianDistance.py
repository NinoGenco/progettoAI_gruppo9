import numpy as np
from KNNAlgorithm.CalculateDistance.Strategy.DistanceStrategy import DistanceStrategy

"""Questa classe implementa il calcolo della distanza euclidea tra due punti x1 e x2.
Non utilizziamo la radice quadrata perchè ci interessa solo ordinare le distanze,
senza trovare i valori esatti. Inoltre così facendo eliminiamo un'operazione
computazionale costosa per il processore."""

class EuclideanDistance(DistanceStrategy):

    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:

        # Eseguo un controllo dimensionale per evitare che numpy applichi un
        # broadcasting errato su vettori di lunghezza diversa.
        if x1.shape != x2.shape:
            raise ValueError("I vettori sono incompatibili perchè hanno dimensioni diverse.")

        return np.sum(pow(x1-x2, 2))