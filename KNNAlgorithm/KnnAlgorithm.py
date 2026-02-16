from collections import Counter
import numpy as np
import random

from KNNAlgorithm.CalculateDistance.Factory.DistanceFactory import DistanceFactory

class KnnAlgorithm:

    """Questa classe gestisce l'intera logica dell'algoritmo K-Nearest Neighbors (KNN). Si occupa di configurare il
    modello, inizializzare i parametri presenti, memorizzare i dati di training e compiere predizioni su nuovi dati,
    basate sulla vicinanza geometrica dei k vicini."""

    def __init__(self, k: int, metric_name: str = 'euclidean'):

        """Costruttore della classe. Inizializza i parametri fondamentali del modello e seleziona la strategia di
        distanza adeguata.

        Parametri: k: Numero di vicini da considerare (deve essere > 0).
                   metric_name: Nome della metrica di distanza da utilizzare."""

        # Validazione per assicurare che il parametro k sia coerente.
        if k <= 0:
            raise ValueError("Il valore di k deve essere un intero positivo.")

        self.X_train = None  # Feature set di addestramento
        self.y_train = None  # Etichette di addestramento
        self.k = k  # Numero di vicini

        # Utilizzo della Factory per istanziare la strategia corretta.
        self.distance_strategy = DistanceFactory.get_distance_metric(metric_name)

    def fit(self, X, y):

        """Esegue la memorizzazione dei dati di addestramento. Converte gli input in array per garantire efficienza nei
        calcoli vettoriali.

        Parametri: X: Il dataset delle Feature di training.
                   y: Il vettore dei Target di training corrispondente."""

        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):

        """Effettua la classificazione, ovvero la predizione delle etichette, per un insieme di nuovi dati. Itera su
        ogni campione del test set e calcola la classe più probabile.

        Parametri: X_test: Insieme dei dati da classificare.

        Risultati: Array contenente le etichette predette per ogni campione X_test."""

        # Verifico che il modello sia stato addestrato prima di provare a predire.
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Errore: chiamare 'fit(X, y)' prima di 'predict(X_test)'.")

        X_test = np.array(X_test)

        # Applico la logica di predizione a tutto il dataset.
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def predict_proba(self, X_test, positive_label=4):

        """Calcola la probabilità di appartenenza alla classe positiva. Invece di restituire la classe, restituisce la
        frazione di vicini che appartengono alla classe positiva. Utile per calcolare curve ROC e AUC.

        Parametri: X_test: Insieme dei dati da classificare.
                   positive_label: L'etichetta che consideriamo positiva, ad esempio 4 per Maligno.

        Risultati: Array che rappresenta la confidenza del modello."""

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Chiamare fit(X,y) prima di predict_proba.")

        X_test = np.array(X_test)
        scores = []

        for x in X_test:

            # Calcolo le distanze verso tutti i punti di training.
            distances = [self.distance_strategy.calculate(x_train, x) for x_train in self.X_train]

            # Identificazione degli indici dei k vicini più prossimi.
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Calcolo della frequenza relativa della classe positiva.
            positives = sum(1 for lab in k_nearest_labels if lab == positive_label)
            scores.append(positives / self.k)

        return np.array(scores)

    def _predict_single(self, x):

        """Gestisce la logica interna per la classificazione di un singolo punto. Calcola le distanze, trova i k vicini
        e applica il voto di maggioranza con gestione casuale dei pareggi.

        Parametri: x: Singolo vettore da classificare.

        Risultati: Restituisce l'etichetta della classe vincitrice."""

        # Calcolo della distanza tra il punto x e tutti i punti memorizzati in X_train.
        distances = [self.distance_strategy.calculate(x_train, x) for x_train in self.X_train]

        # Restituisce gli indici ordinati per distanza crescente, prendiamo i primi k.
        k_indices = np.argsort(distances)[:self.k]

        # Estrae le etichette corrispondenti ai k vicini più prossimi.
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Conteggio delle frequenze delle etichette per determinare la maggioranza.
        vote_counts = Counter(k_nearest_labels)

        # Identificazione del numero massimo di voti.
        max_votes = vote_counts.most_common(1)[0][1]

        # Creo una lista di tutte le etichette con il punteggio massimo.
        winners = [label for label, count in vote_counts.items() if count == max_votes]

        # Se esiste più di un vincitore, ne sceglie uno a caso.
        return random.choice(winners)