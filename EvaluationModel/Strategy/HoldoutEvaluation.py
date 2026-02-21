from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Metrics import metrics_binary
import random


def stratified_train_test_split(X, y, test_size=0.2, shuffle=True, seed=42):
    """
    Funzione: divide il dataset (X, y) in training set e test set in modo STRATIFICATO,
    mantenendo (circa) la stessa proporzione di classi nei due insiemi.

    Parametri:
      - X (Sequence): lista/array dei campioni (features).
      - y (Sequence): lista/array delle label (classi), una per ciascun campione di X.
      - test_size (float): frazione di dataset da assegnare al test set (0 < test_size < 1).
      - shuffle (bool): se True, rimescola gli indici (prima per classe e poi globalmente).
      - seed (int): seme per rendere riproducibile lo shuffle.

    Ritorno:
      - X_train (List): campioni assegnati al training set.
      - X_test (List): campioni assegnati al test set.
      - y_train (List): label corrispondenti ai campioni di training.
      - y_test (List): label corrispondenti ai campioni di test.
    """

    # Validazione
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size deve essere un float tra 0 e 1 (es. 0.2)")
    if len(X) != len(y):
        raise ValueError("X e y devono avere la stessa lunghezza")
    if len(X) < 2:
        raise ValueError("Servono almeno 2 campioni")

    n = len(X)
    # solo classificazione binaria: ci devono essere 2 classi
    labels = list(set(y))
    if len(labels) != 2:
        raise ValueError("Holdout stratificato supporta solo classificazione binaria (2 classi).")

    # Generatore pseudo-casuale locale per rendere deterministiche le operazioni di shuffle
    rnd = random.Random(seed)

    # separa gli indici per classe
    idx_by_class = {lab: [] for lab in labels}
    for i, lab in enumerate(y):
        idx_by_class[lab].append(i)

    # Separa gli indici del dataset per classe: permette di assegnare quote di test per classe
    if shuffle:
        for lab in labels:
            rnd.shuffle(idx_by_class[lab])

    # numero totale di test sample
    n_test_total = int(round(n * test_size))

    # test e train devono essere entrambi non vuoti (almeno 1 nel test e 1 nel train)
    n_test_total = max(1, min(n_test_total, n - 1))

    # assegna un numero di test per classe proporzionale
    counts = {lab: len(idx_by_class[lab]) for lab in labels}

    # Quota di test per classe proporzionale alla numerosità della classe
    n_test = {}
    for lab in labels:
        n_test[lab] = int(round(counts[lab] * test_size))

    # Se possibile, garantisce almeno 1 elemento per classe nel test (utile per ROC/AUC)
    # Condizione: devono esserci almeno 2 elementi in quella classe e almeno 2 test totali
    for lab in labels:
        if counts[lab] >= 2 and n_test_total >= 2:
            n_test[lab] = max(1, n_test[lab])

    # la somma delle quote per classe deve tornare esattamente a n_test_total
    current = sum(n_test.values())

    # Se mancano elementi, li assegna alla classe con più elementi residui disponibili
    while current < n_test_total:
        lab = max(labels, key=lambda l: counts[l] - n_test[l])
        # lascia almeno 1 al training (se possibile)
        if n_test[lab] < counts[lab] - 1:
            n_test[lab] += 1
            current += 1
        else:
            break

    # Se eccedono elementi, li rimuove dalla classe che ne ha di più nel test
    while current > n_test_total:
        lab = max(labels, key=lambda l: n_test[l])
        # Proprietà: se si cerca la binarietà nel test, non scendere sotto 1 (quando possibile)
        min_allowed = 1 if (counts[lab] >= 2 and n_test_total >= 2) else 0
        if n_test[lab] > min_allowed:
            n_test[lab] -= 1
            current -= 1
        else:
            break

    # Costruisce gli indici di test/train selezionando i primi k per il test e i restanti per il train
    test_idx = []
    train_idx = []

    for lab in labels:
        k = n_test[lab]
        test_part = idx_by_class[lab][:k]
        train_part = idx_by_class[lab][k:]

        # Proprietà: se possibile, ogni classe deve comparire anche nel training
        # (altrimenti il modello non può imparare quella classe)
        if len(train_part) == 0 and len(test_part) > 0:
            train_part = [test_part.pop()]

        test_idx.extend(test_part)
        train_idx.extend(train_part)

    # Shuffle finale per evitare che i campioni siano raggruppati per classe nel train/test
    if shuffle:
        rnd.shuffle(train_idx)
        rnd.shuffle(test_idx)

    # Costruzione effettiva degli insiemi
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    # Nota: se il dataset è troppo piccolo o estremamente sbilanciato,
    # può ancora capitare che il test contenga una sola classe → AUC non definibile.
    return X_train, X_test, y_train, y_test


class HoldoutEvaluation(EvaluationStrategy):
    """
    Classe: implementa la strategia di valutazione Holdout STRATIFICATA.
    Rappresentazione interna:
      - non mantiene stato persistente; usa variabili locali in evaluate().
      - produce un singolo split train/test (a differenza del K-Fold che produce più esperimenti)
    """

    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        """
                Metodo: esegue la valutazione del modello tramite Holdout stratificato.

                Parametri:
                  - model: classificatore (k-NN) con metodi fit(), predict() e predict_proba().
                  - X (Sequence): features del dataset.
                  - y (Sequence): label del dataset.
                  - k_neighbors (int): numero di vicini (k) usati dal k-NN.
                  - kwargs:
                      * test_size (float): percentuale del dataset da usare come test (default 0.2).
                      * shuffle (bool): se True rimescola i dati prima dello split (default True).
                      * seed (int): seme per rendere riproducibile lo split (default 42).
                      * positive_label (int): label considerata positiva per ROC/AUC e confusion (default 4).

                Ritorno:
                  - result (dict): dizionario con metodo, parametri, label reali/predette e metriche calcolate.
        """
        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", True)
        seed = kwargs.get("seed", 42)
        positive_label = kwargs.get("positive_label", 4)

        n = len(X)

        # Validazione
        if n != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")
        if n < 2:
            raise ValueError("Servono almeno 2 campioni per fare holdout")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors deve essere > 0")

        # Stratified split: mantiene la distribuzione delle classi tra train e test
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, seed=seed
        )

        # k non può superare il numero di campioni disponibili nel training set
        if k_neighbors > len(X_train):
            raise ValueError("k_neighbors non può essere maggiore del numero di campioni nel training set")

        # Configura e addestra il modello sul training set
        model.k = k_neighbors
        model.fit(X_train, y_train)

        # Predice le label per i campioni nel test set
        y_pred = model.predict(X_test)

        # una predizione per ogni elemento del test set
        if len(y_pred) != len(y_test):
            raise RuntimeError("predict() deve ritornare una label per ogni sample di test")

        # Calcola gli score continui necessari per ROC/AUC
        y_score = model.predict_proba(X_test, positive_label=positive_label)

        # Calcolo metriche: in holdout c'è un solo esperimento, quindi per_run e mean coincidono
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