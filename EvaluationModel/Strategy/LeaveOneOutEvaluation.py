# in testing
# LOO = K-Fold, se n_splits = n

# CONCETTUALMENTE: (LEGGI COMMENTO SOTTO)

'''
La Leave-One-Out cross-validation è stata implementata come caso particolare della K-Fold validation,
impostando il numero di fold pari al numero di campioni.
Per chiarezza architetturale, è stata comunque definita una strategia dedicata LeaveOneOutEvaluation,
che riutilizza internamente la logica della K-Fold.
'''

from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Strategy.KFoldEvaluation import KFoldEvaluation

class LeaveOneOutEvaluation(EvaluationStrategy):
    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        return KFoldEvaluation().evaluate(
            model, X, y,
            k_neighbors=k_neighbors,
            n_splits=len(X),
            shuffle=False,
            seed=kwargs.get("seed", 42),
            positive_label=kwargs.get("positive_label", 4),
        )
