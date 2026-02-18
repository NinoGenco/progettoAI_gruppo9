import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl
from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm
from EvaluationModel.Factory.EvaluationFactory import EvaluationFactory

def visualizza_predizioni(knn_model, X, y, train_ratio=0.7, seed=42):

    """Questa funzione esegue un test visivo delle performance del modello, dividendo il dataset in training e test set,
    addestrando il modello e stampando un confronto tabellare tra le classi reali e quelle predette.

    Parametri: knn_model: Istanza dell'algoritmo K-NN, già configurata con k e metrica, da testare.
               X: Dataframe contenente le Features.
               y: Series contenente il Target.
               train_ratio: Numero float che indica la percentuale di dati da usare per il training, in questo caso il 70%.
               seed: Numero intero per inizializzare il generatore di numeri casuali."""

    print("\nTEST VISIVO (Reale vs Predetto)\n")

    # Reset indici.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)

    # Creo una lista di indici numerici e li mescolo casualmente.
    num_righe = len(X)
    indici = np.arange(num_righe)

    np.random.seed(seed)
    np.random.shuffle(indici)

    # Applico gli indici mescolati ai dati.
    X = X.iloc[indici]
    y = y.iloc[indici]

    # Calcolo a che punto tagliare il dataset.
    punto_di_taglio = int(num_righe * train_ratio)

    # Eseguo la divisione tra training e test set.
    X_train = X.iloc[:punto_di_taglio]
    y_train = y.iloc[:punto_di_taglio]
    X_test = X.iloc[punto_di_taglio:]
    y_test = y.iloc[punto_di_taglio:]

    # Controllo per evitare errori se i dati sono pochi.
    if len(X_test) == 0 or len(X_train) == 0:
        print("Errore: Dataset troppo piccolo per la divisione!")
        return

    print("Distribuzione y_train:", y_train.value_counts().to_dict())
    print("Distribuzione y_test :", y_test.value_counts().to_dict())

    # Addestro il modello sui dati di train e faccio le previsioni.
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)

    # Creo un nuovo DataFrame per mostrare i risultati in modo ordinato.
    results = pd.DataFrame()
    results["Reale"] = y_test.values
    results["Predetto"] = predictions

    # Confronto i valori per capire se la predizione è giusta.
    lista_esiti = []
    for reale, predetto in zip(results["Reale"], results["Predetto"]):
        if reale == predetto:
            lista_esiti.append("CORRETTO")
        else:
            lista_esiti.append("ERRATO")

    results["Esito"] = lista_esiti
    print(results.to_string())

def salva_risultati_csv(results, method_name, filename="report_performance.csv"):

    """Questa funzione si occupa di prendere i risultati dell'output e salvarli in un nuovo file CSV.

    Parametri: results: Dizionario contenente le metriche calcolate ed i parametri usati.
               method_name: Stringa che identifica la tecnica di evaluation utilizzata.
               filename: Il percorso del file CSV su cui scrivere."""

    # Estraggo le metriche ed i parametri dal dizionario dei risultati. In assenza di metriche restituisco un messaggio.
    metrics = results.get('mean', {})
    params = results.get('params', {})
    if not metrics:
        print("Nessuna metrica da salvare.")
        return

    #Creo un dizionario vuoto, per poi inserire tutti i dati da salvare.
    dati_riga = {}

    dati_riga["Method"] = method_name

    for chiave, valore in params.items():  #Parametri
        dati_riga[chiave] = valore

    for chiave, valore in metrics.items():  #Metriche
        dati_riga[chiave] = valore

    # Trasformo il dizionario in un DataFrame, sarà di una sola riga.
    df = pd.DataFrame([dati_riga])

    # Salvo su file.
    try:
        file_esiste = os.path.isfile(filename)

        df.to_csv(
            filename,
            mode='a',
            header=not file_esiste,
            index=False,
            sep=';',
            decimal=','
        )
        print(f"\n[FILE] Performance salvate in: '{filename}'")

    except Exception as e:
        print(f"\n[ERRORE] Salvataggio fallito: {e}")

def salva_plot_confusion_matrix(results, method_name):

    """Questa funzione prende i valori della matrice di confusione (TP, TN, FP, FN) e crea un grafico. Esso viene poi
    salvato come immagine PNG nella cartella 'plots'."""

    # Controllo se i dati sono presenti.
    dati = results.get('mean', {})
    if not dati:
        return

    # Recupero i 4 valori fondamentali. Se non ci sono, uso 0 come default.
    tp = dati.get("tp", 0)
    tn = dati.get("tn", 0)
    fp = dati.get("fp", 0)
    fn = dati.get("fn", 0)

    # Creo la matrice 2x2.
    matrice = np.array([
        [tn, fp],     # Reali Negativi (TN, FP)
        [fn, tp]      # Reali Positivi (FN, TP)
    ])

    # Creo il grafico.
    plt.figure(figsize=(6, 5))
    plt.imshow(matrice, cmap='Blues', alpha=0.8)

    # Aggiungo i numeri al centro di ogni quadrato.
    for i in range(2):
        for j in range(2):
            valore = matrice[i, j]

            # Scrivo il testo al centro del quadrato
            plt.text(j, i, str(valore), ha='center', va='center', fontsize=14, color='black')

    # Titolo e definizione degli assi.
    plt.title(f"Matrice di Confusione - {method_name}")
    plt.xlabel("Classe Predetta")
    plt.ylabel("Classe Reale")

    # Sostituisco i numeri 0 e 1 con le etichette "Benigno" e "Maligno".
    etichette = ['Negativo (2)', 'Positivo (4)']
    plt.xticks([0, 1], etichette)
    plt.yticks([0, 1], etichette)

    # Leggenda
    plt.colorbar()

    # Gestisco il salvataggio del file.
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Creo un nome file con l'orario per non sovrascrivere quelli vecchi.
    orario = time.strftime("%H%M%S")
    nome_file = f"plots/cm_{method_name}_{orario}.png"

    try:
        plt.savefig(nome_file)
        print(f"[PLOT] Grafico salvato correttamente: {nome_file}")
    except Exception as e:
        print(f"Errore nel salvare il grafico: {e}")
    finally:
        plt.close()

def main():

    """Funzione principale del programma. Gestisce il flusso di esecuzione."""

    print("--- FASE 1: Preprocessing ---")

    # Definisco il percorso del file.
    dataset_path = os.path.join("dati", "version_1.csv")

    # Controllo se il file esiste.
    if not os.path.exists(dataset_path):
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        return

    # Istanzio il preprocessor e pulisco i dati.
    preprocessor = PreprocessorImpl()

    try:
        print(f"Elaborazione del file: {dataset_path} ...")
        X, y = preprocessor.preprocess(dataset_path)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(int)

        print("\n=== DATASET PREPROCESSATO COMPLETO ===")

        print("\nFEATURE MATRIX X:")
        print(X.to_string(index=True))

        print("\nTARGET y:")
        print(y.to_string(index=True))

        print("\nPreprocessing completato con successo!")
        print(f"Dimensioni Feature: {X.shape}")
        print(f"Dimensioni Target:  {y.shape[0]}")
        print(f"Feature estratte: {list(X.columns)}")

    except Exception as e:
        print(f"Errore durante il preprocessing: {e}")
        return

    print("\n--- FASE 2: ALGORITMO K-NN ---")

    try:
        k_input = input("Inserisci k (invio per default=3): ")
        k = int(k_input) if k_input.strip() else 3
    except ValueError:
        print("Valore non valido. Uso k=3.")
        k = 3

    knn_model = KnnAlgorithm(k=k, metric_name="euclidean")
    print(f"\nModello KNN creato (k={k}, Metrica='euclidean')")

    visualizza_predizioni(knn_model, X, y, train_ratio=0.7, seed=42)

    print("\n--- FASE 3: EVALUATION ---")

    print("Scegli la strategia di valutazione:")
    print("1. Holdout")
    print("2. K-Fold Cross Validation")
    print("3. Leave-One-Out")

    scelta = input("\nScegliere una di queste tecniche digitando il codice corrispondente: ")

    method_map = {
        "1": "holdout",
        "2": "kfold",
        "3": "loo"
    }

    # Se la scelta non è valida, usa holdout di default
    method_name = method_map.get(scelta.strip(), "holdout")

    # Dizionario per i parametri extra da passare a evaluate()
    evaluation_kwargs = {}

    if method_name == "holdout":
        while True:
            try:
                p_input = input("Inserisci la % di dati per il Training Set (es. 70 per 70%): ")
                valore = float(p_input)

                if 1 <= valore <= 99:
                    # Convertiamo da 70 a 0.7 perché le classi solitamente lavorano con 0.X
                    evaluation_kwargs["test_size"] = 1.0 - (valore / 100.0)
                    print(f"(Impostato: {valore:.0f}% Training - {100 - valore:.0f}% Test)")
                    break
                else:
                    print("Il valore deve essere compreso tra 1 e 99.")
            except ValueError:
                print("Inserire un numero valido.")

    if method_name == "kfold":
        while True:
            try:
                # Chiediamo K, ovvero il numero di Folds (n_splits):
                k_folds_input = input("Inserisci un numero intero di Folds (K ≥ 2) per la Cross Validation (es. K=5,K=10,...): ")
                n_splits = int(k_folds_input)
                evaluation_kwargs["n_splits"] = n_splits
                print(f"(Impostato K-Fold con {n_splits} divisioni)")

            except ValueError:
                print("Valore non valido. Inserisci un numero intero.")

    elif method_name == "loo":
        print("Leave-One-Out selezionato. (Il numero di fold K sarà uguale al numero di campioni).")

    print(f"Esecuzione valutazione: {method_name.upper()} con k={k}...")

    try:
        eval_strategy = EvaluationFactory.create(method_name)

        # Convertiamo X e y in array.
        X_np = X.values
        y_np = y.values

        # Esecuzione valutazione passando il k dato in input dall'utente.
        results = eval_strategy.evaluate(knn_model, X_np, y_np, k_neighbors=k)

        # Chiamo le funzioni di salvataggio.
        salva_risultati_csv(results, method_name)
        salva_plot_confusion_matrix(results, method_name)

        print("\n=== RISULTATI VALUTAZIONE ===")
        # Recupero il dizionario con le medie delle metriche.
        medie = results.get('mean', {})

        if medie:
            print("\nQuale metrica vuoi analizzare?")
            print("1. Accuracy")
            print("2. Error Rate")
            print("3. Sensitivity")
            print("4. Specificity")
            print("5. Geometric Mean")
            print("6. AUC")

            scelta_metrica = input("Inserire il numero corrispondente (1-6): ")

            valore = None
            nome_visualizzato = ""

            if scelta_metrica == "1":
                valore = medie.get("accuracy")
                nome_visualizzato = "ACCURACY"
            elif scelta_metrica == "2":
                valore = medie.get("sensitivity")
                nome_visualizzato = "SENSITIVITY"
            elif scelta_metrica == "3":
                valore = medie.get("specificity")
                nome_visualizzato = "SPECIFICITY"
            elif scelta_metrica == "4":
                valore = medie.get("auc")
                nome_visualizzato = "AUC"
            else:
                print("Metrica non trovata, ti mostro l'Accuracy.")
                valore = medie.get("accuracy")
                nome_visualizzato = "ACCURACY"

            print(f"\n >>> {nome_visualizzato}: {valore} <<< \n")

        else:
            print("Nessun risultato calcolato.")

    except Exception as e:
        print(f"\n[ERRORE CRITICO]: {e}")
        # Stampo l'errore completo per capire cosa è successo
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()