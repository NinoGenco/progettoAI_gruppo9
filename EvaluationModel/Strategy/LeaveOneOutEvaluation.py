# in testing
# LOO = K-Fold, se K = n_splits = n  E ogni fold ha 1 elemento in test

# con n_splits = n e distribuzione round-robin,
# per ogni i vale i % n = i, quindi ogni fold riceve esattamente un indice
# è come fare i = 0,1,2,...,n-1 /n , il resto è i -> ogni numero finisce in un fold nuovo

# CONCETTUALMENTE: (LEGGI COMMENTO SOTTO)

"""

La Leave-One-Out cross-validation è implementata come caso particolare della K-Fold,
impostando il numero di fold pari al numero totale di campioni (n_splits = n).

Tuttavia, a differenza della K-Fold standard, le metriche finali non vengono
calcolate come media fold-level, ma come metriche globali aggregate
(micro-average) su tutte le predizioni.

Questa scelta è motivata dal fatto che, in LOO, ogni fold contiene un solo
campione di test; di conseguenza alcune metriche (es. AUC fold-level)
non sono definibili o risultano numericamente instabili.

"""

from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Strategy.KFoldEvaluation import KFoldEvaluation
from EvaluationModel.Metrics import metrics_binary, confusion_binary, safe_div

class LeaveOneOutEvaluation(EvaluationStrategy):

    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:

        # K-Fold con n_splits = n
        # Con LOO non serve shuffle: ogni campione verrà testato comunque una volta

        result = KFoldEvaluation().evaluate(
            model, X, y,
            k_neighbors=k_neighbors,
            n_splits=len(X),
            shuffle=False,
            seed=kwargs.get("seed", 42),
            positive_label=kwargs.get("positive_label", 4),
        )

        # una predizione per ogni campione (perché ogni campione è stato testato una volta)
        y_true = result["y_true"]
        y_pred = result["y_pred"]

        # Calcola confusion sull’insieme totale dei test
        # (che coincide con tutto il dataset testato una volta ciascuno)

        tp, tn, fp, fn = confusion_binary(y_true, y_pred, positive_label=kwargs.get("positive_label", 4))


        # METRICHE GLOBALI
        # calcolate su n predizioni totali, non su fold da 1

        acc = safe_div(tp + tn, tp + tn + fp + fn)
        err = 1.0 - acc
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        gmean = (sens * spec) ** 0.5

        # AUC globale, riusa l'AUC globale già calcolata in KFoldEvaluation

        auc_global = result["mean"]["auc"]

        return {
            "method": "loo",
            "params": {
                "k": k_neighbors,
                "n_splits": len(X)
            },
            "mean": {
                "accuracy": acc,
                "error_rate": err,
                "sensitivity": sens,
                "specificity": spec,
                "gmean": gmean,
                "auc": auc_global,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            }
        }
