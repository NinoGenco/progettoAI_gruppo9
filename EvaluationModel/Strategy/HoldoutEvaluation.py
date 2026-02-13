from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Metrics import metrics_binary
import random


def stratified_train_test_split(X, y, test_size=0.2, shuffle=True, seed=42):
    """
    Divide (X, y) in train/test in modo STRATIFICATO:
    mantiene (circa) la stessa proporzione di classi in train e test.

    Garantisce, se possibile, che nel test set ci siano entrambe le classi.
    """

    # Validazioni base
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size deve essere un float tra 0 e 1 (es. 0.2)")
    if len(X) != len(y):
        raise ValueError("X e y devono avere la stessa lunghezza")
    if len(X) < 2:
        raise ValueError("Servono almeno 2 campioni")

    n = len(X)
    labels = list(set(y))
    if len(labels) != 2:
        raise ValueError("Holdout stratificato supporta solo classificazione binaria (2 classi).")

    rnd = random.Random(seed)

    # separa gli indici per classe
    idx_by_class = {lab: [] for lab in labels}
    for i, lab in enumerate(y):
        idx_by_class[lab].append(i)

    # shuffle per classe
    if shuffle:
        for lab in labels:
            rnd.shuffle(idx_by_class[lab])

    # numero totale di test sample
    n_test_total = int(round(n * test_size))
    n_test_total = max(1, min(n_test_total, n - 1))

    # assegna un numero di test per classe proporzionale
    counts = {lab: len(idx_by_class[lab]) for lab in labels}

    # quota proporzionale
    n_test = {}
    for lab in labels:
        n_test[lab] = int(round(counts[lab] * test_size))

    # aggiusta per garantire almeno 1 per classe nel test (se possibile)
    for lab in labels:
        if counts[lab] >= 2 and n_test_total >= 2:
            n_test[lab] = max(1, n_test[lab])

    # correggi per far tornare la somma a n_test_total
    current = sum(n_test.values())
    while current < n_test_total:
        # aggiungi 1 alla classe con più elementi residui
        lab = max(labels, key=lambda l: counts[l] - n_test[l])
        if n_test[lab] < counts[lab] - 1:  # lascia almeno 1 al train
            n_test[lab] += 1
            current += 1
        else:
            break

    while current > n_test_total:
        # togli 1 dalla classe con più test allocati, senza scendere sotto 1 se serve binarietà
        lab = max(labels, key=lambda l: n_test[l])
        min_allowed = 1 if (counts[lab] >= 2 and n_test_total >= 2) else 0
        if n_test[lab] > min_allowed:
            n_test[lab] -= 1
            current -= 1
        else:
            break

    # costruisci test/train idx
    test_idx = []
    train_idx = []

    for lab in labels:
        k = n_test[lab]
        test_part = idx_by_class[lab][:k]
        train_part = idx_by_class[lab][k:]

        # garanzia: train deve avere almeno 1 elemento per classe se possibile
        if len(train_part) == 0 and len(test_part) > 0:
            # sposta uno dal test al train
            train_part = [test_part.pop()]
        test_idx.extend(test_part)
        train_idx.extend(train_part)

    # shuffle finale degli indici (opzionale) per non avere blocchi per classe
    if shuffle:
        rnd.shuffle(train_idx)
        rnd.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    # Se ancora il test ha una sola classe (dataset troppo piccolo/sbilanciato), AUC non sarà definita.
    # Ma almeno abbiamo fatto il massimo possibile.
    return X_train, X_test, y_train, y_test


class HoldoutEvaluation(EvaluationStrategy):
    """
    Implementazione della strategia di valutazione Holdout STRATIFICATA.
    """

    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", True)
        seed = kwargs.get("seed", 42)
        positive_label = kwargs.get("positive_label", 4)

        n = len(X)
        if n != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")
        if n < 2:
            raise ValueError("Servono almeno 2 campioni per fare holdout")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors deve essere > 0")

        # Stratified split
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, seed=seed
        )

        if k_neighbors > len(X_train):
            raise ValueError("k_neighbors non può essere maggiore del numero di campioni nel training set")

        model.k = k_neighbors
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if len(y_pred) != len(y_test):
            raise RuntimeError("predict() deve ritornare una label per ogni sample di test")

        y_score = model.predict_proba(X_test, positive_label=positive_label)

        # Qui non dovrebbe più esplodere l'AUC nella maggior parte dei casi,
        # perché lo split è stratificato.
        metrics = metrics_binary(y_test, y_pred, y_score, positive_label=positive_label)

        return {
            "method": "holdout",
            "params": {
                "k_neighbors": k_neighbors,
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
