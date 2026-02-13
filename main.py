import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl
from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm
from EvaluationModel.Factory.EvaluationFactory import EvaluationFactory

def visualizza_predizioni(knn_model, X, y, train_ratio=0.7, seed=42):
    """
    Test visivo Reale vs Predetto.
    Holdout semplificato (train_ratio train, resto test) con SHUFFLE per evitare bias.
    """
    print("\n------------------------------------------------------------")
    print("                 TEST VISIVO (Reale vs Predetto)")
    print("------------------------------------------------------------")

    # Reset indici + cast target
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)

    # Shuffle mantenendo allineamento X/y
    data = X.copy()
    data["__y__"] = y.values
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    X = data.drop(columns="__y__")
    y = data["__y__"]

    # Split
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    if len(X_test) == 0 or len(X_train) == 0:
           print("Dataset troppo piccolo per dividere in train/test!")
           return

    print("Distribuzione y_train:", y_train.value_counts().to_dict())
    print("Distribuzione y_test :", y_test.value_counts().to_dict())

    # Fit + Predict
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)

    # Visualizzazione risultati
    results = pd.DataFrame({
        "Reale": y_test.values,
        "Predetto": predictions
    })
    results["Esito"] = results.apply(
         lambda row: "CORRETTO" if row["Reale"] == row["Predetto"] else "ERRATO",
         axis=1
    )

    print(results)
    print("------------------------------------------------------------\n")


def main():
    print("--- FASE 1: Preprocessing ---")

    dataset_path = os.path.join("dati", "toy_dataset_1.csv")

    if not os.path.exists(dataset_path):
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        return

    preprocessor = PreprocessorImpl()

    try:
        print(f"Elaborazione del file: {dataset_path} ...")
        X, y = preprocessor.preprocess(dataset_path)

        # PERCHE' X e y vengono resettati ?


        # reset_index serve prima di fare split (iloc)
        # astype(int) serve prima di confrontare etichette (==

        # se lo mettessi dopo:
        # lo split sarebbe già fatto
        # rischieresti mismatch o risultati ambigui -> Così eviti possibili casini in holdout/k-fold

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(int)

        print("\n=== DATASET PREPROCESSATO COMPLETO ===")

        print("\n--- FEATURE MATRIX X ---")
        print(X.to_string(index=True))

        print("\n--- TARGET y ---")
        print(y.to_string(index=True))

        print("Preprocessing completato con successo!")
        print(f" -> Dimensioni Feature (X): {X.shape}")
        print(f" -> Dimensioni Target (y):  {y.shape[0]}")
        print(f" -> Feature estratte: {list(X.columns)}")

    except Exception as e:
        print(f"Errore durante il preprocessing: {e}")
        return  # importante: se fallisce, non andare al KNN

    print("\n--- FASE 2: KNNAlgorithm ---")

    try:
        k_input = input("    -> Inserisci k (numero vicini, invio per default=3): ")
        k = int(k_input) if k_input.strip() else 3
    except ValueError:
        print("       ! Valore non valido. Uso k=3.")
        k = 3

    knn_model = KnnAlgorithm(k=k, metric_name="euclidian")
    print(f"    -> Modello KNN creato (K={k}, Metrica='euclidian')")

    visualizza_predizioni(knn_model, X, y, train_ratio=0.7, seed=42)

    print("\n--- FASE 3: EVALUATION ---")

    print("Scegli la strategia di valutazione:")
    print("1. Holdout")
    print("2. K-Fold Cross Validation")
    print("3. Leave-One-Out")

    scelta = input(" -> Scegliere una di queste tecniche digitando il codice corrispondente: ")

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
                p_input = input("    -> Inserisci la % di dati per il Training Set (es. 70 per 70%): ")
                valore = float(p_input)

                if 1 <= valore <= 99:
                    # Convertiamo da 70 a 0.7 perché le classi solitamente lavorano con 0.X
                    evaluation_kwargs["train_size"] = valore / 100.0
                    print(f"       (Impostato: {valore:.0f}% Training - {100 - valore:.0f}% Test)")
                    break
                else:
                    print("       ! Il valore deve essere compreso tra 1 e 99.")
            except ValueError:
                print("       ! Inserire un numero valido.")

    if method_name == "kfold":
        while True:
            try:
                # Chiediamo K (Folds) solo per la K-Fold
                k_folds_input = input("    -> Inserisci il numero di Folds (K) per la Cross Validation (es. 5, 10): ")
                n_splits = int(k_folds_input)

                if n_splits < 2:
                    print("       ! Il numero di folds deve essere almeno 2. Riprova.")
                else:
                    evaluation_kwargs["n_splits"] = n_splits
                    print(f"       (Impostato K-Fold con {n_splits} divisioni)")
                    break
            except ValueError:
                print("       ! Valore non valido. Inserisci un numero intero.")

    elif method_name == "loo":
        print("    -> Leave-One-Out selezionato. (Il numero di fold K sarà uguale al numero di campioni).")

    print(f"    -> Esecuzione valutazione: {method_name.upper()} con K={k}...")

    try:
        # Creazione della strategia tramite Factory
        eval_strategy = EvaluationFactory.create(method_name)

        # IMPORTANTE: Convertiamo X e y in array numpy o liste.
        # Le strategie di valutazione usano indici posizionali (es. X[i]),
        # che non funzionano direttamente sui DataFrame Pandas.

        X_np = X.values
        y_np = y.values

        # Esecuzione valutazione passando il K dato in input dall'utente
        results = eval_strategy.evaluate(knn_model, X_np, y_np, k_neighbors=k)

        print("\n=== RISULTATI VALUTAZIONE ===")
        # Recupera il dizionario con le medie delle metriche
        metrics = results.get('mean', {})

        if metrics:
            # --- MENU INTERATTIVO PER SCELTA METRICA ---
            print("\nQuale metrica vuoi analizzare?")
            print("1. Accuracy")
            print("2. Error Rate")
            print("3. Sensitivity")
            print("4. Specificity")
            print("5. Geometric Mean")
            print("6. AUC")

            scelta_metric = input(" -> Inserisci il numero (1-6, invio per Accuracy): ")

            # Mappa: Input Utente -> Chiave del dizionario results
            metrics_map = {
                "1": "accuracy",
                "2": "error_rate",
                "3": "sensitivity",
                "4": "specificity",
                "5": "gmean",
                "6": "auc"
            }

            # Default su 'accuracy' se l'input non è valido o vuoto
            selected_key = metrics_map.get(scelta_metric.strip(), "accuracy")

            # --- VISUALIZZAZIONE SOLO DELLA METRICA SCELTA ---
            valore_scelto = metrics.get(selected_key)

            print(f"\n{'-' * 40}")
            if valore_scelto is not None:
                # Formattazione a 4 cifre decimali se è float
                if isinstance(valore_scelto, float):
                    print(f" >>> {selected_key.upper()}: {valore_scelto:.4f} <<<")
                else:
                    print(f" >>> {selected_key.upper()}: {valore_scelto} <<<")
            else:
                print(f" >>> {selected_key.upper()}: N/A (Non disponibile) <<<")
            print(f"{'-' * 40}\n")

        else:
            print("Nessun risultato prodotto.")

    except Exception as e:
        print(f"\nERRORE CRITICO durante la valutazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()