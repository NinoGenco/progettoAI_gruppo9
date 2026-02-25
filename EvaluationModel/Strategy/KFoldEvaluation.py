from EvaluationModel.Strategy.EvaluationStrategy import EvaluationStrategy
from EvaluationModel.Metrics import metrics_binary, confusion_binary, safe_div
import random


def k_fold_split_indices(n, n_splits=5, shuffle=True, seed=42):
    """
    Funzione: costruisce una suddivisione K-Fold del dataset sotto forma di liste di indici

    Parametri:
      - n (int): numero totale di campioni nel dataset.
      - n_splits (int): numero di fold (K) da generare.
      - shuffle (bool): se True, rimescola gli indici prima di distribuirli nei fold.
      - seed (int): seme per rendere riproducibile lo shuffle

    Ritorno:
      - folds (List[List[int]]): lista di K liste, ciascuna contenente gli indici dei campioni del fold.
    """

    # n_splits = 5 di default è standard practice

    # controlli di validità:
    if n_splits < 2:
        raise ValueError("n_splits deve essere >= 2")
    if n_splits > n:
        raise ValueError("n_splits non può essere maggiore del numero di campioni")

    indices = list(range(n))

    # Se Selezionato, rimescola gli indici in modo riproducibile
    if shuffle:
        random.Random(seed).shuffle(indices)

    # Inizializza K contenitori vuoti (uno per fold)
    # Assegna alla lista di shuffled indici, 'idx' una posizione 'i' e tramite il resto (i % n_splits) fa Round-Robin
    # Garantisce fold di dimensione molto simile (differenza al più 1)
    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(indices):
        folds[i % n_splits].append(idx)

    return folds


def mean_metrics(per_run):
    """
    Funzione: calcola la media delle metriche ottenute su più run/fold.

    Parametri:
      - per_run (List[dict]): lista di dizionari, uno per fold, con le metriche calcolate in quel fold.
    Ritorno:
      - out (dict): dizionario con le metriche mediate. Se una metrica è None in tutti i fold, resta None.
    """

    # se non ci sono run, non è possibile calcolare nessuna media
    if not per_run:
        return {}

    # Le chiavi sono uguali per tutti i dizionari (uno per fold)
    keys = per_run[0].keys()
    out = {}

    # Per ogni metrica, fa la media ignorando i None (es. auc non definita in fold che hanno solo una tipologia di tumori, es: [4,4,4,4])
    for k in keys:
        vals = [d[k] for d in per_run if d.get(k) is not None] # ignora i None
        out[k] = (sum(vals) / len(vals)) if vals else None

    return out



class KFoldEvaluation(EvaluationStrategy):
    """
    Classe: implementa la strategia di valutazione K-Fold Cross Validation.
    Rappresentazione interna:
      - non mantiene stato persistente; usa variabili locali durante evaluate().
      - produce:
          * per_run: metriche calcolate fold per fold
          * mean: metriche aggregate (media fold-level e/o globali, in base alla logica finale)
    """

    def evaluate(self, model, X, y, k_neighbors: int, **kwargs) -> dict:
        """
                Metodo: esegue la valutazione del modello tramite K-Fold Cross Validation.
                Parametri:
                  - model: classificatore (qui un k-NN) con metodi fit(), predict() e predict_proba().
                  - X (Sequence): features del dataset (lista/array di campioni).
                  - y (Sequence): label del dataset (lista/array di classi).
                  - k_neighbors (int): parametro k del k-NN (numero di vicini usati dal modello).
                  - kwargs:
                      * n_splits (int): numero di fold K (default 5).
                      * shuffle (bool): se True rimescola il dataset prima di splittare (default True).
                      * seed (int): seme di shuffle per riproducibilità (default 42).
                      * positive_label (int): etichetta considerata "positiva" per confusion/ROC (default 4).
                Ritorno:
                  - result (dict): dizionario con metodo, parametri, predizioni globali, metriche per fold e aggregate.
        """

        # Parametri di configurazione della cross-validation (K = n_splits)
        n_splits = kwargs.get("n_splits", 5)
        shuffle = kwargs.get("shuffle", True)
        seed = kwargs.get("seed", 42)
        positive_label = kwargs.get("positive_label", 4)

        n = len(X)

        # controlli di validità:
        if n != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")
        if n < 2:
            raise ValueError("Servono almeno 2 campioni")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors deve essere un intero positivo maggiore di zero")
        if n_splits < 2:
            raise ValueError("n_splits deve essere >= 2")

        # Costruisce la lista dei fold come indici che andranno nel test
        folds = k_fold_split_indices(n, n_splits=n_splits, shuffle=shuffle, seed=seed)

        # Accumulatori:
        # - per_run contiene le metriche calcolate per ogni fold
        # - *_all accumulano tutte le predizioni complessive (utile per AUC globale e per micro-average)

        per_run = []
        y_true_all = []
        y_pred_all = []
        y_score_all = []

        # Loop principale: ogni iterazione usa un fold come test, gli altri come training
        for fold_idx in range(n_splits):


            # Indici di test = fold corrente
            test_idx = folds[fold_idx]

            # Indici di training = unione di tutti i fold tranne quello corrente
            # Proprietà: train_idx e test_idx sono disgiunti e coprono tutti i campioni
            train_idx = [i for j in range(n_splits) if j != fold_idx for i in folds[j]]

            # Costruisce i set in base agli indici
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            # Guardia: in ogni fold k non può eccedere il numero di punti disponibili in training
            if k_neighbors > len(X_train):

                '''Guardia fondamentale, soprattutto per LOO:
                in LOO len(X_train) = n-1, k non può superare i dati disponibili
                '''
                # per fare il voto devi avere almeno k punti nel train, importante soprattutto in LOO o dataset piccoli.
                raise ValueError(
                    f"k_neighbors={k_neighbors} non può essere > del training set nel fold {fold_idx} "
                    f"(len(X_train)={len(X_train)})"
                )

            # Imposta il parametro k del modello k-NN
            model.k = k_neighbors

            # Addestramento sul training set del fold
            model.fit(X_train, y_train)

            # Predizione delle classi sul test set del fold
            y_pred = model.predict(X_test)

            # Proprietà: il modello deve produrre una predizione per ogni sample di test
            if len(y_pred) != len(y_test):
                raise RuntimeError("predict() deve ritornare una label per ogni sample di test")

            # Calcola uno score continuo per ROC/AUC:
            # Score continuo che rappresenti quanto il modello “spinge” verso la classe positiva -> misura quanto il modello è convinto
            # necessario per ROC/AUC, è un valore numerico ordinabile che misura il grado di appartenenza
            # alla classe positiva (proporzione di vicini positivi nel KNN), permettendo di variare la soglia decisionale
            # e costruire la curva ROC


            y_score = model.predict_proba(X_test, positive_label=positive_label)

            # ---------- accumulo globale (serve per AUC globale, e per LOO) ----------

            # predizioni di tutto il dataset

            # Accumulo globale: serve per metriche globali e AUC su tutto il dataset

            # Aggiunge tutte le y vere di questo fold a una lista globale.
            y_true_all.extend(y_test)

            # Normalizza y_pred e y_score a liste Python (supporta anche numpy array)
            y_pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
            y_pred_all.extend(y_pred_list)
            y_score_list = y_score.tolist() if hasattr(y_score, "tolist") else list(y_score)
            y_score_all.extend(y_score_list)

            # ---------- metriche fold-level SENZA AUC se fold non binario ----------

            # Fold non binario: niente ROC/AUC
            tp, tn, fp, fn = confusion_binary(y_test, y_pred_list, positive_label=positive_label)

            acc = safe_div(tp + tn, tp + tn + fp + fn)
            err = 1.0 - acc
            sensitivity = safe_div(tp, tp + fn)  # TPR
            specificity = safe_div(tn, tn + fp)  # TNR
            gmean = (sensitivity * specificity) ** 0.5

            if len(set(y_test)) == 2:
                m_full = metrics_binary(y_test, y_pred_list, positive_label=positive_label)
                auc_fold = m_full["auc"]
            else:
                auc_fold = None

            per_run.append({
                "accuracy": acc,
                "error_rate": err,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "gmean": gmean,
                "auc": auc_fold,  # None in LOO, numero in KFold normale (quasi sempre)
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            })

        # FINE LOOP INTERNO

        # ---------- mean fold-level ----------
        # mean_metrics() ignora i None, L’AUC finale diventa la media solo dei fold validi
        mean_m = mean_metrics(per_run)

        # ---------- AUC globale su tutte le predizioni accumulate (sempre definibile se y_true è binario)----------
        # global_m serve a calcolare le metriche su tutte le predizioni accumulate insieme, non fold per fold
        # serve per avere una AUC globale stabile:
        #  PERCHE' ?

        # 1) alcuni fold hanno NONE (quasi sempre in LOO)
        # 2) AUC non è una metrica lineare → la media delle AUC dei fold non è uguale all’AUC sull’intero dataset
        # 3) Nei fold piccoli (es. LOO) l’AUC fold-level è instabile o impossibile
        global_m = metrics_binary(y_true_all, y_pred_all, y_score_all, positive_label=positive_label)

        # tutte le altre metriche sono additive sulla confusion matrix, auc no
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