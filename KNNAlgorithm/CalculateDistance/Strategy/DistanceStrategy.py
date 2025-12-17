from abc import abstractmethod, ABC
import numpy as np

class DistanceStrategy(ABC):

    @abstractmethod
    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        pass