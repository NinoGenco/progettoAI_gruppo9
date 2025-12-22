from abc import abstractmethod, ABC
import numpy as np

class DistanceStrategy(ABC):

    """ Questa interfaccia astratta definisce il calcolo della distanza tra vettori. """

    @abstractmethod
    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:

        """ Calcola la distanza tra due vettori.

        :param x1: Primo vettore (campione di addestramento).
        :param x2: Secondo vettore (punto di test).
        :return: Valore float che rappresenta la distanza tra i due vettori."""

        pass