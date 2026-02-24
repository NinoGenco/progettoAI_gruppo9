from abc import abstractmethod, ABC
import numpy as np

class DistanceStrategy(ABC):

    """ Questa classe astratta funge da interfaccia per il Design Pattern 'Strategy', definendo una modalità standard che
    tutte le strategie di calcolo della distanza dovranno rispettare e garantenfo che ogni metrica implementi il proprio metodo."""

    @abstractmethod
    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:

        """ Calcola la distanza tra due vettori n-dimensionali.

        Parametri:
                  x1 -> Primo vettore numerico (Solitamente un campione estratto dal Training Set).
                  x2 -> Secondo vettore numerico (Solitamente un singolo campione estratto dal Test Set da classificare).

        Risultati -> Valore numerico float che rappresenta la distanza calcolata tra x1 e x2."""

        pass