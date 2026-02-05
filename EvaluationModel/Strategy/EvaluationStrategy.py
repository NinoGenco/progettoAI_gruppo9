# INTERFACCIA (Strategy Pattern)

# Responsabilità:
# Definire un contratto comune per tutte le strategie di valutazione.
# Ogni strategia (Holdout, K-Fold, Leave-One-Out, ecc.)
# deve rispettare questa interfaccia.

from abc import ABC, abstractmethod

class EvaluationStrategy(ABC):
    """
        Classe astratta che rappresenta una strategia di valutazione del modello.

        Questa classe NON implementa alcuna logica concreta.
        Serve solo a definire "che cosa deve fare" una strategia di evaluation,
        non "come lo deve fare".
        """
    @abstractmethod
    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        """
        Metodo astratto che ogni strategia di valutazione deve implementare.

        Parametri:
        - model: il modello di classificazione (k-NN)
        - X: feature matrix (lista o array di feature)
        - y: vettore delle label
        - k_neighbors: numero di vicini da usare nel k-NN
        - **kwargs: parametri specifici della strategia
                    (es. test_size per Holdout, K per K-Fold, ...)

        ritorna un dizionario, in base  alla strategia specificato dall'utente
        """
        pass

