import os
import sys
import pandas as pd

# Assicuriamoci che Python trovi i moduli (opzionale ma consigliato)
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
        k_input = input("    -> Inserisci K (numero vicini, invio per default=3): ")
        k = int(k_input) if k_input.strip() else 3
    except ValueError:
        print("       ! Valore non valido. Uso K=3.")
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

        # In base alla strategia, 'mean' contiene il riassunto delle metriche
        metrics = results.get('mean', {})

        if metrics:
            for metric, value in metrics.items():
                if value is not None:
                    # Formattiamo i float, lasciamo invariati gli interi (come TP, TN...)
                    if isinstance(value, float):
                        print(f" - {metric}: {value:.4f}")
                    else:
                        print(f" - {metric}: {value}")
                else:
                    print(f" - {metric}: N/A")
        else:
            print("Nessun risultato prodotto.")

    except Exception as e:
        print(f"\nERRORE CRITICO durante la valutazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()