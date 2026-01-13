from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Metrics import metrics_binary, confusion_binary, safe_div
import random


def k_fold_split_indices(n, n_splits=5, shuffle=True, seed=42):
    """
    Crea i fold come liste di indici.
    Distribuzione bilanciata (round-robin) dopo eventuale shuffle.
    """
    if n_splits < 2:
        raise ValueError("n_splits deve essere >= 2")
    if n_splits > n:
        raise ValueError("n_splits non può essere maggiore del numero di campioni")

    indices = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(indices)

    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(indices):
        folds[i % n_splits].append(idx)

    return folds


def mean_metrics(per_run):
    """
    Media delle metriche sui fold.
    - ignora valori None (es. auc fold-level in LOO)
    - se per una chiave tutti i valori sono None -> ritorna None
    """
    if not per_run:
        return {}

    keys = per_run[0].keys()
    out = {}

    for k in keys:
        vals = [d[k] for d in per_run if d.get(k) is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None

    return out



class KFoldEvaluation(EvaluationStrategy):
    """
    Strategia K-Fold Cross Validation.

    kwargs:
      - n_splits: int (default 5)
      - shuffle: bool (default True)
      - seed: int (default 42)
      - positive_label: int (default 4)
    """

    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        n_splits = kwargs.get("n_splits", 5)
        shuffle = kwargs.get("shuffle", True)
        seed = kwargs.get("seed", 42)
        positive_label = kwargs.get("positive_label", 4)

        n = len(X)
        if n != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")
        if n < 2:
            raise ValueError("Servono almeno 2 campioni")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors deve essere un intero positivo maggiore di zero")
        if n_splits < 2:
            raise ValueError("n_splits deve essere >= 2")

        folds = k_fold_split_indices(n, n_splits=n_splits, shuffle=shuffle, seed=seed)

        per_run = []
        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for fold_idx in range(n_splits):
            test_idx = folds[fold_idx]
            train_idx = [i for j in range(n_splits) if j != fold_idx for i in folds[j]]

            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            if k_neighbors > len(X_train):
                raise ValueError(
                    f"k_neighbors={k_neighbors} non può essere > del training set nel fold {fold_idx} "
                    f"(len(X_train)={len(X_train)})"
                )

            model.k = k_neighbors

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if len(y_pred) != len(y_test):
                raise RuntimeError("predict() deve ritornare una label per ogni sample di test")

            y_score = model.predict_proba(X_test, positive_label=positive_label)

            # ---------- accumulo globale (serve per AUC globale, e per LOO) ----------
            y_true_all.extend(y_test)
            y_pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
            y_pred_all.extend(y_pred_list)
            y_score_list = y_score.tolist() if hasattr(y_score, "tolist") else list(y_score)
            y_score_all.extend(y_score_list)

            # ---------- metriche fold-level SENZA AUC se fold non binario ----------
            # calcolo confusion + metriche base sempre definite
            tp, tn, fp, fn = confusion_binary(y_test, y_pred_list, positive_label=positive_label)

            acc = safe_div(tp + tn, tp + tn + fp + fn)
            err = 1.0 - acc
            sensitivity = safe_div(tp, tp + fn)
            specificity = safe_div(tn, tn + fp)
            gmean = (sensitivity * specificity) ** 0.5

            # AUC fold-level solo se nel fold ci sono entrambe le classi
            if len(set(y_test)) == 2:
                m_full = metrics_binary(y_test, y_pred_list, y_score_list, positive_label=positive_label)
                auc_fold = m_full["auc"]
            else:
                auc_fold = None  # non definita per fold non binari (LOO)

            per_run.append({
                "accuracy": acc,
                "error_rate": err,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "gmean": gmean,
                "auc": auc_fold,   # None in LOO, numero in KFold normale (quasi sempre)
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            })

        # ---------- mean fold-level ----------
        mean_m = mean_metrics(per_run)

        # ---------- AUC globale (sempre definita se dataset è binario) ----------
        global_m = metrics_binary(y_true_all, y_pred_all, y_score_all, positive_label=positive_label)
        mean_m["auc"] = global_m["auc"]

        return {
            "method": "kfold",
            "params": {
                "k": k_neighbors,
                "n_splits": n_splits,
                "shuffle": shuffle,
                "seed": seed,
                "positive_label": positive_label,
            },
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "per_run": per_run,
            "mean": mean_m,
        }
