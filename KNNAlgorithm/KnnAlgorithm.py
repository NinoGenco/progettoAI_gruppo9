from collections import Counter
import numpy as np
import random

from KNNAlgorithm.CalculateDistance.Factory.DistanceFactory import DistanceFactory

class KnnAlgorithm:

    """Questa classe gestisce l'intera logica dell'algoritmo K-Nearest Neighbors (KNN) per la classificazione binaria.
    Si occupa di memorizzare i dati di addestramento e compiere predizioni, sia come classe che come probabilità,
    su nuovi dati, basandosi sulla vicinanza geometrica dei k vicini."""

    def __init__(self, k: int, metric_name: str = 'euclidean'):

        """Costruttore della classe. Inizializza i parametri fondamentali del modello e seleziona la strategia di
        distanza adeguata.

        Parametri:
                  k -> Numero di vicini da considerare, deve essere un valore positivo e maggiore di 0.
                  metric_name -> Stringa indicante il nome della metrica da utilizzare per calcolare le distanze."""

        # Gestisco il caso in cui k non sia coerente con quanto scritto sopra, ovvero se non fosse > 0.
        if k <= 0:
            # Blocco l'esecuzione del programma stampando un messaggio di errore.
            raise ValueError("Il valore di k deve essere un intero positivo.")

        self.X_train = None  # Inizializzo a none la variabile che conterrà le Feature del train test.
        self.y_train = None  # Inizializzo a none la variabile che conterrà i Target del train test.
        self.k = k  # Memorizzo il numero di vicini.

        # Utilizzo il Factory per la creazione dell'oggetto che calcolerà le distanze.
        self.distance_strategy = DistanceFactory.get_distance_metric(metric_name)

    def fit(self, X, y):

        """Memorizza i dati di addestramento all'interno dello stato interno del modello. Inoltre converte gli input in
        array per garantire efficienza nei calcoli vettoriali.

        Parametri:
                   X -> Array delle Feature di training.
                   y -> Array dei Target di training."""

        # Gestisco il caso in cui X e y non hanno lo stesso numero di righe, infatti ogni feature deve avere la propria etichetta.
        if len(X) != len(y):
            # Blocco l'esecuzione del programma stampando un messaggio di errore.
            raise ValueError("X e y devono avere la stessa lunghezza.")

        self.X_train = np.array(X)  # Converte la lista delle feature in array.
        self.y_train = np.array(y)  # Converte la lista dei target in array.

    def predict(self, X_test):

        """Effettua la predizione su un intero set di dati di test. Itera su ogni campione del test set e calcola la
        classe più probabile.

        Parametri:
                  X_test -> Insieme dei dati da classificare.

        Risultati -> Array contenente l'etichetta predetta per ciascun campione di test."""

        # Gestisco il caso in cui i dati di addestramento non siano stati caricati, quindi il modello noe è addestrato prima di provare a predire.
        if self.X_train is None:
            # Blocco l'esecuzione del programma stampando un messaggio di errore.
            raise RuntimeError("Errore: Devi addestrare il modello con fit() prima di predire.")

        X_test = np.array(X_test) # Converto il dataset di test in array.
        predictions = [] # Creo una lista vuota in cui verranno inserite le previsioni fatte.

        # Applico la logica di predizione, iterando su ogni riga del test set.
        for x in X_test:
            result = self._predict_single(x) # Chiamo il metodo predict_single per predire la classe del punto considerato.
            predictions.append(result) # Aggiungo la previsione calcolata alla lista predictions.

        return np.array(predictions) # Restituisce l'elenco completo delle previsioni come array.

    def predict_proba(self, X_test, positive_label=4):

        """Calcola la probabilità di appartenenza alla classe positiva per un set di dati. Invece di restituire la classe,
        restituisce la frazione di vicini che appartengono alla classe positiva. Utile per calcolare curve ROC e AUC.

        Parametri:
                  X_test -> Insieme dei dati da classificare.
                  positive_label -> Etichetta considerata come target positivo, di default 4 per Maligno.

        Risultati -> Array contenente la frazione di vicini positivi per ogni campione. Per frazione di vicini si intende
                     il rapporto tra il numero di vicini positivi e il numero totale di vicini."""

        # Gestisco il caso in cui i dati di addestramento non siano stati caricati, quindi il modello noe è addestrato prima di provare a predire.
        if self.X_train is None:
            # Blocco l'esecuzione del programma stampando un messaggio di errore.
            raise RuntimeError("Errore: Devi addestrare il modello con fit() prima di predire.")

        X_test = np.array(X_test) # Converto il dataset di test in array.
        scores = [] # Creo una lista vuota in cui verranno inserite le probabilità.

        # Applico questo metodo iterado su ogni riga del test set.
        for x in X_test:
            k_nearest_labels = self._get_k_nearest_labels(x) # Chiamo il metodo get_k_nearest_labels per ottenere le etichette dei k vicini più prossimi al punto considerato.

            positive_count = 0 # Inizializzo un contatore per ricordare il numero di vicini appartenenti alla classe positiva.

            # Itero una ad una le etichette dei k vicini appena trovati.
            for label in k_nearest_labels:
                if label == positive_label: # Verifico se l'etichetta corrente corrisponde a quella considerata come classe positiva.
                    positive_count += 1 # Incremento il contatore di 1 se il vicino appartiene alla classe positiva.

            scores.append(positive_count / self.k) # Calcolo il rapporto tra i vicini positivi ed il totale k, aggiungendo questo score alla lista.

        return np.array(scores) # Restituisce l'elenco completo degli score come array.

    def _get_k_nearest_labels(self, x):
        """Calcola la distanza e restituisce le etichette dei k vicini più prossimi.

        Parametri:
                  x -> Vettore numerico che rappresenta un campione di test da confrontare.

        Risultati -> Lista contenente le etichette dei k campioni di addestramento più vicini."""

        distances = [] # Creo una lista vuota in cui verranno inserite tutte le distanze calcolate.

        # Itera su tutti i punti memorizzati nel dataset di addestramento.
        for x_train in self.X_train:
            dist = self.distance_strategy.calculate(x_train, x) # Chiamo lo strategy per il calcolo della distanza tra i due punti.
            distances.append(dist) # Aggiungo il valore della distanza calcolata alla lista distances.

        etichette_vicini = []  # Creo una lista vuota in cui verranno inserite le etichette vicine.

        # Iteriamo l'operazione k volte.
        for i in range(self.k):
            min_dist = min(distances)  # Calcolo la distanza più piccola.
            indice_min = distances.index(min_dist)  # Identifichiamo in quale posizione si trovi.

            etichetta = self.y_train[indice_min]  # Prendo l'etichetta in quella stessa posizione.
            etichette_vicini.append(etichetta)  # Aggiunto tale etichetta alla lista etichette_vicini.

            distances[indice_min] = float('inf') # Metto la distanza a infinito così al prossimo giro non la peschiamo più.

        return etichette_vicini  # Restituisce le etichette trovate.

    def _predict_single(self, x):

        """Gestisce la logica interna per la classificazione di un singolo punto, con gestione casuale dei pareggi.

        Parametri:
                  x -> Vettore numerico che rappresenta un campione di test da classificare.

        Risultati -> Restituisce l'etichetta della classe vincitrice."""

        k_nearest_labels = self._get_k_nearest_labels(x) # Sfrutto il metodo get_k_nearest_label per ottenere le etichette dei vicini.

        vote_counts = Counter(k_nearest_labels) # Conto le occorrenze di ciascuna etichetta creando un dizionario di frequenze.

        max_votes = vote_counts.most_common(1)[0][1] # Estraggo il conteggio più alto tra i voti ricevuti, ovvero il primo elemento della lista.

        winners = [] # Creo una lista in cui inserire la classe vincente, ne potrebbero esserci molteplici.

        # Esamino il conteggio totale di voti per ogni singola etichetta.
        for label, count in vote_counts.items():
            if count == max_votes: # Verifico se l'etichetta corrente ha raggiunto il punteggio massimo.
                winners.append(label) # Se la condizione if è verificata inserisco tale etichetta alla lista winners.

        return random.choice(winners) # Se esiste più di un vincitore, ne sceglie uno a caso.