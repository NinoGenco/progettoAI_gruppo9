import numpy as np
from KNNAlgorithm.CalculateDistance.Strategy.DistanceStrategy import DistanceStrategy

class EuclidianDistance(DistanceStrategy):

    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(pow(x1-x2, 2))