import sys
import os
import pandas as pd
import numpy as np

# Permette a Python di trovare i package 'KNNAlgorithm' e 'Preprocessing'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl

def manual_train_test_split(X, y, test_ratio=0.2):

    """ Divide il dataset in Training e Test set, mescolando gli indici."""

    if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
    if not isinstance(y, pd.Series): y = pd.Series(y)

    num_samples = len(X)
    indices = np.arange(num_samples)

    # Shuffle (con seed per riproducibilità)
    np.random.seed(42)
    np.random.shuffle(indices)

    # Calcolo indici di taglio
    split_index = int(num_samples * (1 - test_ratio))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Restituzione dati splittati
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


def run_integration_test():
    print("=============================================================")
    print("TEST INTEGRATO: PREPROCESSOR IMPL + KNN")
    print("=============================================================")

    # FASE 1: PREPROCESSING

    print("\n--- FASE 1: CARICAMENTO E PULIZIA DATI ---")

    # Percorso al file CSV
    data_path = os.path.join(current_dir, "../dati", "version_1.csv")

    X, y = None, None

    if os.path.exists(data_path):
        try:
            print(f"File trovato: {data_path}")

            # ISTANZIAMO L'IMPLEMENTAZIONE.
            preprocessor = PreprocessorImpl()

            # ESEGUIAMO IL PREPROCESSING.
            X, y = preprocessor.preprocess(data_path)

            print(f"Preprocessing completato con successo.")
            print(f"Shape Features (X): {X.shape}")
            print(f"Shape Target (y):   {y.shape}")
            print(f"Colonne Features: {list(X.columns)}")

        except Exception as e:
            print(f"Errore critico nel Preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return  # Interrompiamo il test se il preprocessing fallisce
    else:
        print(f"File non trovato in: {data_path}")
        print("Impossibile eseguire il test senza il dataset.")
        return

    # FASE 2: SPLIT TRAIN/TEST

    print("\n--- FASE 2: SPLIT DATASET (80/20) ---")

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=0.2)

    print(f"Training Set: {len(X_train)} campioni")
    print(f"Test Set: {len(X_test)} campioni")

    # FASE 3: KNN ALGORITHM
    print("\n--- FASE 3: ESECUZIONE KNN ---")

    try:
        #Inizializzazione
        k_val = 5
        knn = KnnAlgorithm(k=k_val, metric_name='euclidian')
        print(f"Algoritmo inizializzato (k={k_val}).")

        #Addestramento
        print("Addestramento in corso...")
        knn.fit(X_train, y_train)
        print("Addestramento completato.")

        #Predizione (Su un piccolo campione per verifica)
        sample_size = min(20, len(X_test))  # Testiamo max 20 righe per pulizia output
        print(f"Calcolo predizioni su {sample_size} campioni di test...")

        X_test_sample = X_test.iloc[:sample_size]
        y_test_sample = y_test.iloc[:sample_size]

        predictions = knn.predict(X_test_sample)

        #Verifica Risultati
        print("\n=== RISULTATI DEL TEST ===")
        y_true = y_test_sample.values

        matches = np.sum(predictions == y_true)
        accuracy = (matches / len(predictions)) * 100

        print(f"Predizioni: {predictions}")
        print(f"Reali: {y_true}")
        print(f"\n Accuratezza su questo campione: {accuracy:.2f}%")

        if accuracy > 80:
            print("Ottimo! Il sistema sembra funzionare correttamente.")

    except Exception as e:
        print(f"\n ERRORE NEL KNN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_integration_test()