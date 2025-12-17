from abc import ABC
from KNNAlgorithm.CalculateDistance.Strategy.EuclidianDistance import EuclidianDistance

class DistanceFactory(ABC):
    @staticmethod

    def get_distance_metric(metric_name: str) -> 'DistanceStrategy':

        clean_name = metric_name.lower().strip()

        if clean_name == 'euclidian':
            return EuclidianDistance()
        else:
            raise ValueError(f"La metrica '{metric_name}' non è supportata o non esiste.")