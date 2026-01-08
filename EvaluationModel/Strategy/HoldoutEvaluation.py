from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
import random


def train_test_split(X, y, test_size=0.2, shuffle=True, seed=42):

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size deve essere un float tra 0 e 1 (es. 0.2)")

    if len(X) != len(y):
        raise ValueError("X e y devono avere la stessa lunghezza")

    if len(X) < 2:
        raise ValueError("Servono almeno 2 campioni")

    n = len(X)
    indices = list(range(n))

    if shuffle:
        random.Random(seed).shuffle(indices)

    split = int(round(n * (1 - test_size)))
    split = max(1, min(split, n - 1))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


def _confusion_binary(y_true, y_pred, positive_label=4):
    """
    Confusion matrix binaria:
    TP, TN, FP, FN rispetto a positive_label.
    """
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_label and p == positive_label:
            tp += 1
        elif t != positive_label and p != positive_label:
            tn += 1
        elif t != positive_label and p == positive_label:
            fp += 1
        elif t == positive_label and p != positive_label:
            fn += 1
    return tp, tn, fp, fn


def _safe_div(num, den):
    return num / den if den != 0 else 0.0

def _roc_curve(y_true, y_score, positive_label=4):
    labels = sorted(set(y_true))
    if len(labels) != 2:
        raise ValueError("ROC supporta solo classificazione binaria")

    if positive_label not in labels:
        raise ValueError("positive_label non presente in y_true")

    negative_label = labels[0] if labels[1] == positive_label else labels[1]

    thresholds = sorted(set(y_score), reverse=True)
    thresholds = [float("inf")] + thresholds + [float("-inf")]

    fpr_list, tpr_list = [], []

    for thresh in thresholds:
        tp = tn = fp = fn = 0

        for t, s in zip(y_true, y_score):
            pred = positive_label if s >= thresh else negative_label

            if t == positive_label and pred == positive_label:
                tp += 1
            elif t == negative_label and pred == negative_label:
                tn += 1
            elif t == negative_label and pred == positive_label:
                fp += 1
            else:
                fn += 1

        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return fpr_list, tpr_list


"""
    L’AUC è stata calcolata a partire dalla curva ROC, 
    utilizzando come score la proporzione di vicini appartenenti 
    alla classe positiva nel classificatore k-NN.
"""

def _auc(fpr, tpr):

    """
    Trapezoidal rule.
    Assumes fpr is sorted increasingly.
    """
    auc = 0.0
    for i in range(1, len(fpr)):
        x1, x2 = fpr[i - 1], fpr[i]
        y1, y2 = tpr[i - 1], tpr[i]
        auc += (x2 - x1) * (y1 + y2) / 2
    return auc




def _metrics_binary(y_true, y_pred, y_score, positive_label=4):
    tp, tn, fp, fn = _confusion_binary(y_true, y_pred, positive_label)

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    err = 1.0 - acc
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    gmean = (sensitivity * specificity) ** 0.5

    fpr, tpr = _roc_curve(y_true, y_score, positive_label)

    # ✅ ordina per FPR crescente prima di integrare
    pairs = sorted(zip(fpr, tpr))
    if len(pairs) < 2:
        auc = 0.0
    else:
        fpr_sorted, tpr_sorted = zip(*pairs)
        auc = _auc(fpr_sorted, tpr_sorted)

    return {
        "accuracy": acc,
        "error_rate": err,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "gmean": gmean,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class HoldoutEvaluation(EvaluationStrategy):
    """
    Holdout validation: 1 run con split train/test.

    kwargs supportati:
      - test_size: float (default 0.2)
      - shuffle: bool (default True)
      - seed: int|None (default 42)
      - positive_label: int (default 4)  # utile per sensitivity/specificity
    """

    def evaluate(self, model, X, y, **kwargs) -> dict:
        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", True)
        seed = kwargs.get("seed", 42)
        positive_label = kwargs.get("positive_label", 4)

        n = len(X)
        if n != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")
        if n < 2:
            raise ValueError("Servono almeno 2 campioni per fare holdout")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )

        # Fit + Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if len(y_pred) != len(y_test):
            raise RuntimeError("predict() deve ritornare una label per ogni sample di test")

        y_score = model.predict_proba(X_test, positive_label=positive_label)
        metrics = _metrics_binary(y_test, y_pred, y_score, positive_label=positive_label)

        # Per holdout: una sola run => per_run ha 1 elemento e mean = metrics
        return {
            "method": "holdout",
            "params": {
                "test_size": test_size,
                "shuffle": shuffle,
                "seed": seed,
                "positive_label": positive_label,
            },
            "y_true": y_test,
            "y_pred": y_pred,
            "per_run": [metrics],
            "mean": metrics,
        }
