# INTERFACCIA

# Responsabilità -> definire un contratto unico:
# dato il k-NN, dati X, y e parametri (k vicini, K fold ecc.), ritorna risultati.

from abc import ABC, abstractmethod

class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        """
        X is the feature matrix
        y is the label vector

        Returns a dict with:
          - y_true: list
          - y_pred: list
          - per_run: list[dict]   (metrics for each fold/run)
          - mean: dict            (average metrics)
        """
        pass

