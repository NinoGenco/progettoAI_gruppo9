# in testing
# LOO = K-Fold, se n_splits = n

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

        # 1️⃣ Esegui K-Fold con n_splits = n
        result = KFoldEvaluation().evaluate(
            model, X, y,
            k_neighbors=k_neighbors,
            n_splits=len(X),
            shuffle=False,
            seed=kwargs.get("seed", 42),
            positive_label=kwargs.get("positive_label", 4),
        )

        y_true = result["y_true"]
        y_pred = result["y_pred"]

        # 2️⃣ Calcola confusion globale
        tp, tn, fp, fn = confusion_binary(
            y_true, y_pred,
            positive_label=kwargs.get("positive_label", 4)
        )

        acc = safe_div(tp + tn, tp + tn + fp + fn)
        err = 1.0 - acc
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        gmean = (sens * spec) ** 0.5

        # 3️⃣ AUC globale
        y_score = result["mean"]["auc"]  # oppure ricalcolala da score se preferisci

        metrics_global = metrics_binary(
            y_true,
            y_pred,
            result["y_pred"],  # meglio usare score reali se disponibili
            positive_label=kwargs.get("positive_label", 4)
        )

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
                "auc": metrics_global["auc"],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            }
        }
