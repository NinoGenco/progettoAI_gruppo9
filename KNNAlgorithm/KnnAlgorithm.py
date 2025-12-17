from collections import Counter
import numpy as np
import random

from KNNAlgorithm.CalculateDistance.Factory.DistanceFactory import DistanceFactory

class KnnAlgorithm:

    """ Implementazione dell'algoritmo K-Nearest Neighbors (KNN).

    Questa classe gestisce il ciclo di vita del classificatore:
    1. Inizializzazione dei parametri (k e metrica).
    2. Addestramento (memorizzazione dei dati).
    3. Predizione basata sulla maggioranza dei voti dei k vicini."""

    def __init__(self, k: int, metric_name: str = 'euclidean'):

        """ Inizializza il classificatore.

        :param k: Numero di vicini da considerare (deve essere > 0).
        :param metric_name: Nome della metrica di distanza (default: 'euclidean')."""

        if k <= 0:
            raise ValueError("Il valore di k deve essere un intero positivo.")

        self.X_train = None
        self.y_train = None
        self.k = k
        self.distance_strategy = DistanceFactory.get_distance_metric(metric_name)

    def fit(self, X, y):

        """ Memorizza i dati di addestramento.

        Converte gli input in array NumPy per garantire efficienza nei calcoli vettoriali.
        :param X: Feature set (matrice di campioni).
        :param y: Target set (vettore delle etichette)."""

        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):

        """ Esegue la predizione su un intero set di dati di test.

        :param X_test: Array di punti da classificare.
        :return: Array NumPy contenente le etichette predette.
        :raises RuntimeError: Se il metodo viene chiamato prima di fit()."""

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Errore: chiamare 'fit(X, y)' prima di 'predict(X_test)'.")

        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):

        """ Logica interna per la classificazione di un singolo punto.

        Calcola le distanze, identifica i k vicini e gestisce eventuali pareggi
        tramite scelta casuale tra i vincitori."""

        distances = [self.distance_strategy.calculate(x_train, x) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        vote_counts = Counter(k_nearest_labels)

        max_votes = vote_counts.most_common(1)[0][1]

        winners = [label for label, count in vote_counts.items() if count == max_votes]

        return random.choice(winners)