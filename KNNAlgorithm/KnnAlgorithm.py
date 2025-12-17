from collections import Counter
import numpy as np
import random

from KNNAlgorithm.CalculateDistance.Factory.DistanceFactory import DistanceFactory

class KnnAlgorithm:

    def __init__(self, k: int, metric_name: str = 'euclidian'):

        if k <= 0:
            raise ValueError("Il valore di k deve essere un intero positivo.")

        self.X_train = None
        self.y_train = None
        self.k = k

        self.distance_strategy = DistanceFactory.get_distance_metric(metric_name)

    def fit(self, X, y):

        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Devi chiamare il metodo 'fit(X, y)' prima di poter fare previsioni.")

        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):

        distances = [self.distance_strategy.calculate(x_train, x) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        vote_counts = Counter(k_nearest_labels)
        max_votes = vote_counts.most_common(1)[0][1]

        winners = [label for label, count in vote_counts.items() if count == max_votes]

        return random.choice(winners)