# EvaluationModel/Factory/EvaluationFactory.py

from EvaluationModel.Strategy.HoldoutEvaluation import HoldoutEvaluation
from EvaluationModel.Strategy.KFoldEvaluation import KFoldEvaluation
from EvaluationModel.Strategy.LeaveOneOutEvaluation import LeaveOneOutEvaluation


class EvaluationFactory:
    """
    Factory per creare strategie di evaluation (Strategy Pattern + Factory).
    """

    _registry = {
        "holdout": HoldoutEvaluation,
        "kfold": KFoldEvaluation,
        "loo": LeaveOneOutEvaluation,
    }

    @staticmethod
    def create(method: str):
        """
        method: "holdout" | "kfold" | "loo"
        """
        if not isinstance(method, str):
            raise TypeError("method deve essere una stringa (es. 'holdout', 'kfold', 'loo')")

        key = method.strip().lower()
        if key not in EvaluationFactory._registry:
            valid = ", ".join(EvaluationFactory._registry.keys())
            raise ValueError(f"Metodo di evaluation non supportato: '{method}'. Valori validi: {valid}")

        return EvaluationFactory._registry[key]()  # istanzia la classe
