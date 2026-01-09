from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Metrics import metrics_binary
import random


def train_test_split(X, y, test_size=0.2, shuffle=True, seed=42):
    """
        Divide (X, y) in train/test secondo la percentuale test_size.

        - test_size: frazione di campioni da destinare al test (0 < test_size < 1)
        - shuffle: se True, mescola gli indici prima dello split
        - seed: rende lo shuffle riproducibile
        """

    # Exceptions' Handling:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size deve essere un float tra 0 e 1 (es. 0.2)")

    if len(X) != len(y):
        raise ValueError("X e y devono avere la stessa lunghezza")

    if len(X) < 2:
        raise ValueError("Servono almeno 2 campioni")

    n = len(X)
    indices = list(range(n)) # lista degli indici [0,1,2,...,n-1]

    # Shuffle opzionale: mescola indici con seed fissato per riproducibilità
    if shuffle:
        random.Random(seed).shuffle(indices)

    # Calcolo punto di split: numero di campioni in train = n*(1-test_size)
    split = int(round(n * (1 - test_size)))

    # Garantisce che train e test abbiano almeno 1 elemento ciascuno
    split = max(1, min(split, n - 1))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test

class HoldoutEvaluation(EvaluationStrategy):
    """
    Implementazione della strategia di valutazione Holdout:
    esegue un singolo split train/test e calcola le metriche sul test.

    kwargs supportati:
      - test_size: float (default 0.2)
      - shuffle: bool (default True)
      - seed: int|None (default 42)
      - positive_label: int (default 4)
    """

    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        """
                Valuta il modello con una singola holdout split.

                Parametri principali:
                - k_neighbors: numero di vicini del k-NN (richiesto dall'interfaccia)

                Parametri opzionali (kwargs):
                - test_size, shuffle, seed, positive_label
        """

        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", True)
        seed = kwargs.get("seed", 42)
        positive_label = kwargs.get("positive_label", 4)

        # Controlli di coerenza sui dati
        n = len(X)
        if n != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")
        if n < 2:
            raise ValueError("Servono almeno 2 campioni per fare holdout")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors deve essere un intero positivo maggiore di zero")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )
        if k_neighbors > len(X_train):
            raise ValueError("k_neighbors non può essere maggiore del numero di campioni nel training set")

        model.k = k_neighbors

        # Addestramento del modello sul train
        model.fit(X_train, y_train)

        # Predizione label sul test
        y_pred = model.predict(X_test)

        # Verifica che la predict ritorni una label per ogni sample di test
        if len(y_pred) != len(y_test):
            raise RuntimeError("predict() deve ritornare una label per ogni sample di test")

        y_score = model.predict_proba(X_test, positive_label=positive_label)
        metrics = metrics_binary(y_test, y_pred, y_score, positive_label=positive_label)

        # Per holdout: una sola run => per_run ha 1 elemento e mean = metrics
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

