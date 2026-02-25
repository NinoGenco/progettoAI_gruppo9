import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl
from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm
from EvaluationModel.Factory.EvaluationFactory import EvaluationFactory

def visualizza_predizioni(knn_model, X, y, train_ratio=0.7, seed=42):

    """Questa funzione esegue un test visivo rapido per dimostrare il funzionamento del modello. Divide il dataset,
    addestra il KNN e stampa a video una tabella comparativa tra la classe reale e quella predetta.

    Parametri:
              knn_model -> Istanza dell'algoritmo KNN da testare.
              X: DataFrame contenente le features.
              y: Series contenente le etichette target.
              train_ratio: Frazione del dataset da usare per il training.
              seed: Comando per la generazione di numeri casuali, garantisce la riproducibilità dello shuffle."""

    print("\nTEST VISIVO (Reale vs Predetto)\n")

    # Resetta gli indici di X e y per evitare problemi di disallineamento durante lo slicing.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)

    num_righe = len(X) # Calcolo il numero totale di campioni nel dataset.
    indici = np.arange(num_righe) # Creo un array di indici da 0 fino al numero di righe totali meno uno.

    np.random.seed(seed) # Imposto il seed per rendere l'ordine casuale ripetibile a ogni esecuzione.
    np.random.shuffle(indici) # Mescolo casualmente l'array degli indici.

    # Riorganizzo X e y applicando gli indici mescolati.
    X = X.iloc[indici]
    y = y.iloc[indici]

    punto_di_taglio = int(num_righe * train_ratio) # Calcolo l'indice di separazione tra set di addestramento e set di test.

    # Divido i dati usando l'indice calcolato.
    X_train = X.iloc[:punto_di_taglio]
    y_train = y.iloc[:punto_di_taglio]
    X_test = X.iloc[punto_di_taglio:]
    y_test = y.iloc[punto_di_taglio:]

    # Gestisco il caso in cui sia il set di test che il set di train siano vuoti.
    if len(X_test) == 0 or len(X_train) == 0:
        # Blocco l'esecuzione del programma stampando un messaggio di errore.
        print("Errore: Dataset troppo piccolo per la divisione!")
        return

    # Mostra a video come sono distribuite le classi nei due split.
    print("Distribuzione y_train:", y_train.value_counts().to_dict())
    print("Distribuzione y_test :", y_test.value_counts().to_dict())

    knn_model.fit(X_train, y_train) # Addestro il modello sui dati di training.
    predictions = knn_model.predict(X_test) # Effettuo la classificazione sui dati di test.

    results = pd.DataFrame() # Creo un nuovo DataFrame vuoto per memorizzare i risultati.
    results["Reale"] = y_test.values # Inserisco la colonna con i valori reali.
    results["Predetto"] = predictions # Inserisco la colonna con i valori predetti.

    # Usa una list comprehension per valutare riga per riga se la previsione è corretta.
    lista_esiti = [
        "CORRETTO" if reale == predetto else "ERRATO"
        for reale, predetto in zip(results["Reale"], results["Predetto"])
    ]

    results["Esito"] = lista_esiti # Aggiungo la colonna con l'esito.
    print(results.to_string()) # Stampo l'intero DataFrame formattato come testo.


def salva_risultati_csv(results, method_name, filename="performances/report_performance.csv"):
    """Questa funzione estrapola le metriche di valutazione generate dall'algoritmo e le salva in modo su un file CSV.
    Se il file è gia esistente allora aggiunge solamente i nuovi dati, altrimenti ne crea uno nuovo.

    Parametri:
               results -> Dizionario contenente i risultati ed i parametri calcolati.
               method_name -> Nome del metodo di valutazione usato.
               filename: Percorso di destinazione in cui salvare o aggiornare il file CSV."""

    cartella = os.path.dirname(filename) # Estraggo il percorso della cartella dal nome del file.

    # Verifico se la cartella di destinazione esiste prima di poter scrivere il file.
    if cartella and not os.path.exists(cartella):
        os.makedirs(cartella) # Se la condizione if non è verificata creo la cartella.

    metrics = results.get('mean', {}) # Recupero il dizionario delle metriche.
    params = results.get('params', {}) # Recupero il dizionario dei parametri.

    # Gestisco il caso in cui non ci fossero dati da salvare.
    if not metrics:
        # Blocco l'esecuzione del programma stampando un messaggio di errore.
        print("Nessuna metrica da salvare.")
        return

    dati_riga = {"Method": method_name} # Creo un dizionario con il nome del metodo per la nuova riga CSV.

    # Aggiungo tutti i parametri impostati al dizionario.
    for chiave, valore in params.items():
        dati_riga[chiave] = valore

    # Aggiungo tutte le metriche calcolate al dizionario.
    for chiave, valore in metrics.items():
        dati_riga[chiave] = valore

    df = pd.DataFrame([dati_riga]) # Converte il dizionario in un DataFrame di una singola riga.

    # Controllo se il file CSV esiste già all'interno del sistema.
    try:
        file_esiste = os.path.isfile(filename)

        df.to_csv(
            filename,
            mode='a', # 'a' sta per append, aggiungo in coda senza cancellare il vecchio contenuto.
            header=not file_esiste, # Scrivo le intestazioni di colonna solo se il file viene creato nuovo.
            index=False, # Ometto l'indice di riga generato.
            sep=';', # Uso il punto e virgola come separatore di colonne.
            decimal=','
        )

        print(f"\n[FILE] Performance salvate in: '{filename}'")

    # Blocco il programma in caso di eventuali errori di scrittura, stampando un messaggio di errore.
    except Exception as e:
        print(f"\n[ERRORE] Salvataggio fallito: {e}")


def salva_plot_confusion_matrix(results, method_name):

    """Questa funzione disegna graficamente la Matrice di Confusione e la salva in formato PNG.

    Parametri:
              results -> Dizionario con i risultati, dal quale vengono estratti i Veri Positivi, Veri Negativi ecc.
              method_name -> Nome della strategia di valutazione, usato per generare il titolo del grafico."""

    dati = results.get('mean', {}) # Estraggo le metriche dal dizionario.

    if not dati: #Se non ci sono dati blocco il programma.
        return

    # Passo i parametri tp, tn, fp, fn necessari per costruire la matrice.
    tp = dati.get("tp", 0) # True Positive.
    tn = dati.get("tn", 0) # True Negative.
    fp = dati.get("fp", 0) # False Positive.
    fn = dati.get("fn", 0) # False Negative.

    # Costruisco la matrice di confusione con dimensione 2x2.
    matrice = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(figsize=(6, 5)) # Inizializzo una figura per il plot con dimensioni specifiche.
    plt.imshow(matrice, cmap='Blues', alpha=0.8) # Specifiche sulle caratteristiche dell'immagine.

    # Doppio ciclo per inserire il valore testuale al centro di ogni cella della matrice.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrice[i, j]),
                     ha='center', va='center', fontsize=14)

    plt.title(f"Matrice di Confusione - {method_name}") # Aggiungo il nome del titolo.
    plt.xlabel("Classe Predetta") # Aggiungo il nome dell'asse x.
    plt.ylabel("Classe Reale") # Aggiungo il nome dell'asse y.

    etichette = ['Negativo (2)', 'Positivo (4)']
    plt.xticks([0, 1], etichette)
    plt.yticks([0, 1], etichette)
    plt.colorbar()

    if not os.path.exists("plots"):
        os.makedirs("plots") # Controlla e crea la cartella 'plots' se mancante.

    orario = time.strftime("%H%M%S")
    nome_file = f"plots/cm_{method_name}_{orario}.png"

    try:
        plt.savefig(nome_file) # Salvo l'immagine.
        print(f"[PLOT] Grafico salvato correttamente: {nome_file}")
    except Exception as e:
        print(f"Errore nel salvare il grafico: {e}")
    finally:
        plt.close()


def main():

    """ Questa funzione gestisce il parsing degli argomenti CLI forniti dall'utente, richiama il Preprocessing, instanzia
    l'algoritmo KNN, esegue le valutazioni tramite Factory Pattern ed infine richiama le funzioni per salvare e visualizzare i report."""

    parser = argparse.ArgumentParser(description="KNN Classification Pipeline")

    # Configurazione di tutte le opzioni accettate da riga di comando, ad esempio --k, --method, etc.
    parser.add_argument("--k", type=int, default=3,
                        help="Numero di vicini per KNN (default=3)")

    parser.add_argument("--method", type=str, default="holdout",
                        choices=["holdout", "kfold", "loo"],
                        help="Metodo di valutazione")

    parser.add_argument("--train", type=float, default=70,
                        help="Percentuale training set (solo per holdout)")

    parser.add_argument("--folds", type=int, default=5,
                        help="Numero di folds (solo per kfold)")

    parser.add_argument("--dataset", type=str,
                        default=os.path.join("dati", "version_1.csv"),
                        help="Percorso dataset")

    args = parser.parse_args() # Convalido e memorizzo gli argomenti forniti dall'utente.

    print("\n--- FASE 1: Preprocessing ---")

    dataset_path = args.dataset

    # Gestisco il caso in cui non è presente il file del dataset nel percorso specificato.
    if not os.path.exists(dataset_path):
        # Blocco il programma stampando un messaggio di errore.
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        return

    preprocessor = PreprocessorImpl() # Chiamo la classe incaricata della pulizia dei dati.

    try:
        print(f"Elaborazione del file: {dataset_path} ...")
        X, y = preprocessor.preprocess(dataset_path) # Estraggo le features (X) ed i target (y) puliti.

        X = X.reset_index(drop=True) # Resetto l'indice per evitare inconsistenze nei successivi accessi per posizione.
        y = y.reset_index(drop=True).astype(int) # Accerto che le label siano numeri interi.

        print("Preprocessing completato!")
        print(f"Feature shape: {X.shape}")

    # Gestisco possibili errori.
    except Exception as e:
        # Blocco il programma stampando un messaggio di errore.
        print(f"Errore durante il preprocessing: {e}")
        return

    print("\n--- FASE 2: KNN ---")

    # Creo il modello KNN usando i parametri dell'utente.
    k = args.k
    knn_model = KnnAlgorithm(k=k, metric_name="euclidean")

    visualizza_predizioni(knn_model, X, y) # Richiamo il test rapido per verificare che l'algoritmo impari.

    print("\n--- FASE 3: Evaluation ---")

    method_name = args.method
    evaluation_kwargs = {} # Creo un dizionario vuoto in cui inserire gli argomenti specifici per la strategia scelta.

    # Imposto i parametri necessari a seconda se l'utente ha scelto Holdout, K-Fold o Leave-One-Out.
    if method_name == "holdout":
        evaluation_kwargs["test_size"] = 1.0 - (args.train / 100.0) # Converto la percentuale in una frazione per il test.
        print(f"Holdout: {args.train}% training")

    elif method_name == "kfold":
        evaluation_kwargs["n_splits"] = args.folds # Passo il numero di fold scelto.
        print(f"KFold: {args.folds} folds")

    elif method_name == "loo":
        print("Leave-One-Out selezionato")

    try:
        # Uso il Factory Pattern per creare la strategia di validazione corretta.
        eval_strategy = EvaluationFactory.create(method_name)

        # Delego l'intera esecuzione dell'esperimento alla strategia estratta dalla Factory.
        results = eval_strategy.evaluate(
            knn_model,
            X.values,
            y.values,
            k_neighbors=k,
            **evaluation_kwargs
        )

        salva_risultati_csv(results, method_name) # Salvo le metriche.
        salva_plot_confusion_matrix(results, method_name) # Salvo il grafico.

        print("\n=== RISULTATI ===")

        medie = results.get('mean', {})
        # Itero sul dizionario delle metriche medie per stamparle a video formattate in maiuscolo.
        for metrica, valore in medie.items():
            print(f"{metrica.upper()}: {valore}")

    # Gestisco errori gravi avvenuti durante il calcolo e la validazione.
    except Exception as e:
        # Blocco il programma stampando un messaggio di errore.
        print(f"\n[ERRORE CRITICO]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()