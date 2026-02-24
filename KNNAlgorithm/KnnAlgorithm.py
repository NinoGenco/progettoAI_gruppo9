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
        self.y_train = None  # Target di addestramento
        self.k = k  # Numero di vicini

        # Utilizzo della Factory per istanziare la strategia corretta.
        self.distance_strategy = DistanceFactory.get_distance_metric(metric_name)

    def fit(self, X, y):

        """Esegue la memorizzazione dei dati di addestramento. Converte gli input in array per garantire efficienza nei
        calcoli vettoriali.

        Parametri: X: Il dataset delle Feature di training.
                   y: Il vettore dei Target di training corrispondente."""

        # Controlliamo se X e y hanno lo stesso numero di righe.
        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):

        """Effettua la predizione su un intero set di dati di test. Itera su ogni campione del test set e calcola la
        classe più probabile.

        Parametri: X_test: Insieme dei dati da classificare.

        Risultati: Vettore con tutte le predizioni effettuate."""

        # Verifico che il modello sia stato addestrato prima di provare a predire.
        if self.X_train is None:
            raise RuntimeError("Errore: Devi addestrare il modello con fit() prima di predire.")

        X_test = np.array(X_test)
        predictions = []

        # Applico la logica di predizione, iterando su ogni riga del test set.
        for x in X_test:
            # Prediciamo la classe per il singolo punto.
            result = self._predict_single(x)
            predictions.append(result)

        return np.array(predictions)

    def predict_proba(self, X_test, positive_label=4):

        """Calcola la probabilità di appartenenza alla classe positiva. Invece di restituire la classe, restituisce la
        frazione di vicini che appartengono alla classe positiva. Utile per calcolare curve ROC e AUC.

        Parametri: X_test: Insieme dei dati da classificare.
                   positive_label: L'etichetta che consideriamo positiva, ad esempio 4 per Maligno.

        Risultati: Array che rappresenta la confidenza del modello."""

        if self.X_train is None:
            raise RuntimeError("Modello non addestrato.")

        X_test = np.array(X_test)
        scores = []

        for x in X_test:
            # Otteniamo direttamente le etichette dei vicini con il nuovo helper
            k_nearest_labels = self._get_k_nearest_labels(x)

            # Contiamo quanti sono 'positivi'.
            positive_count = 0
            for label in k_nearest_labels:
                if label == positive_label:
                    positive_count += 1

            # Calcoliamo la frequenza.
            scores.append(positive_count / self.k)

        return np.array(scores)

    def _get_k_nearest_labels(self, x):
        """Calcola la distanza e restituisce le etichette dei k vicini più prossimi.

        Parametri: x: Vettore numerico che rappresenta un campione di test da confrontare.

        Risultati: Lista contenente le etichette dei k campioni di addestramento più vicini."""
        distances = []
        for x_train in self.X_train:
            dist = self.distance_strategy.calculate(x_train, x)
            distances.append(dist)

        # Troviamo gli indici dei k valori più piccoli.
        k_indices = np.argsort(distances)[:self.k]

        # Restituiamo le etichette di questi vicini.
        return [self.y_train[i] for i in k_indices]

    def _predict_single(self, x):

        """Gestisce la logica interna per la classificazione di un singolo punto, con gestione casuale dei pareggi.

        Parametri: x: Vettore numerico che rappresenta un campione di test da classificare.

        Risultati: Restituisce l'etichetta della classe vincitrice."""

        # Sfruttiamo il metodo helper per ottenere le etichette dei vicini
        k_nearest_labels = self._get_k_nearest_labels(x)

        # Conteggio delle frequenze delle etichette per determinare la maggioranza.
        vote_counts = Counter(k_nearest_labels)

        # Identificazione del numero massimo di voti.
        max_votes = vote_counts.most_common(1)[0][1]

        # Creo una lista di tutte le etichette con il punteggio massimo.
        winners = []
        for label, count in vote_counts.items():
            if count == max_votes:
                winners.append(label)

        # Se esiste più di un vincitore, ne sceglie uno a caso.
        return random.choice(winners)