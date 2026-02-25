"""
Contiene funzioni riusabili per calcolare metriche di classificazione binaria
richieste dal progetto:
- Accuracy
- Error Rate
- Sensitivity (TPR)
- Specificity (TNR)
- G-Mean
- AUC (da curva ROC)

Questo modulo è usato da tutte le strategie di evaluation (Holdout, K-Fold, LOO).
"""
from typing import List, Tuple

# Exception handling to division by zero
def safe_div(num, den):
    # e.g. se tp = fn = 0, per convenzione è 0

    return num / den if den != 0 else 0.0

def confusion_binary(y_true, y_pred, positive_label=4)-> Tuple[int, int, int, int]:
    """
        Calcola la confusion matrix per classificazione binaria,
        definendo "classe positiva" = positive_label.

        """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devono avere la stessa lunghezza")

    tp,tn,fp,fn = 0,0,0,0
    # Scorre vero e predetto in parallelo
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


def roc_curve(y_true, y_score, positive_label=4):
    """
        Costruisce la curva ROC per classificazione binaria.

        - y_score: "confidence/score" che indica quanto il modello è convinto
                   che il campione sia positivo (qui: proporzione di vicini positivi).
        Ritorna:
          - fpr_list: lista False Positive Rate
          - tpr_list: lista True Positive Rate
        """
    if len(y_true) != len(y_score):
        raise ValueError("y_true e y_score devono avere la stessa lunghezza")

    labels = sorted(set(y_true))
    if len(labels) != 2:
        raise ValueError("ROC supporta solo classificazione binaria")

    if positive_label not in labels:
        raise ValueError("positive_label non presente in y_true")

    # Per fare la ROC, è necessario sapere chi è positivo e chi è negativo
    negative_label = labels[0] if labels[1] == positive_label else labels[1]

    # Thresholds:
    # Prende tutti gli score unici (soglie candidate) dal più alto al più basso
    #  +inf e -inf per includere estremi della curva

    thresholds = sorted(set(y_score), reverse=True)
    thresholds = [float("inf")] + thresholds + [float("-inf")]

    # accumula i punti della curva
    fpr_list, tpr_list = [], []

    # Per ogni soglia, trasforma score in predizione binaria e calcola (FPR,TPR)
    for thresh in thresholds:
        tp = tn = fp = fn = 0

        for t, s in zip(y_true, y_score):
            # cambiare soglia cambia quanta roba predici positiva
            pred = positive_label if s >= thresh else negative_label

            if t == positive_label and pred == positive_label:
                tp += 1
            elif t == negative_label and pred == negative_label:
                tn += 1
            elif t == negative_label and pred == positive_label:
                fp += 1
            else:
                fn += 1

        tpr = safe_div(tp, tp + fn)
        fpr = safe_div(fp, fp + tn)

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return fpr_list, tpr_list

# L’AUC viene calcolata dalla curva ROC usando come score
# la proporzione di vicini appartenenti alla classe positiva nel k-NN.

def auc_trapezoid(fpr: List[float], tpr: List[float]) -> float:
    """
    Calcolo AUC con regola del trapezio.
    Assunzione: fpr ordinato in modo crescente, dove i punti sono (x = FPR, y = TPR).
    Immagina un trapezio rettangolo con le basi parallele all'asse verticale di un piano cartesiano
    """
    auc = 0.0
    for i in range(1, len(fpr)):
        x1, x2 = fpr[i - 1], fpr[i] # x2 -x1 è l'altezza del trapezio
        y1, y2 = tpr[i - 1], tpr[i] # y1 e y2 sono le basi
        auc += (x2 - x1) * (y1 + y2) / 2.0
    return auc

def metrics_binary(y_true, y_pred, y_score, positive_label=4):
    """
        Calcola tutte le metriche richieste dal progetto:
        accuracy, error_rate, sensitivity, specificity, gmean, auc + confusion counts.
        """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devono avere la stessa lunghezza")

    if len(y_true) != len(y_score):
        raise ValueError("y_true e y_score devono avere la stessa lunghezza")

    tp, tn, fp, fn = confusion_binary(y_true, y_pred, positive_label)

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    err = 1.0 - acc
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    gmean = (sensitivity * specificity) ** 0.5

    fpr, tpr = roc_curve(y_true, y_score, positive_label)

    # Ordina per FPR crescente prima di integrare
    pairs = sorted(zip(fpr, tpr))
    if len(pairs) < 2:
        auc = 0.0
    else:
        fpr_sorted, tpr_sorted = zip(*pairs)
        auc = auc_trapezoid(list(fpr_sorted), list(tpr_sorted))

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
        "fn": fn
    }