from abc import abstractmethod, ABC
import numpy as np

class DistanceStrategy(ABC):

    """ Questa classe astratta definisce una modalità standard per tutte le strategie di calcolo della distanza.
    Impone alle sottoclassi di implementare il metodo di calcolo specifico."""

    @abstractmethod
    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:

        """ Calcola la distanza tra due vettori n-dimensionali.

        Parametri: x1: Primo vettore numerico (Solitamente un campione del Training Set).
                   x2: Secondo vettore numerico (Solitamente il punto di Test Set da classificare).

        Risultati: Valore numerico float che rappresenta la distanza calcolata tra x1 e x2."""

        pass