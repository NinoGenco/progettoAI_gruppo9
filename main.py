import os
import sys
import pandas as pd

# Assicuriamoci che Python trovi i moduli (opzionale ma consigliato)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiamo l'implementazione concreta
from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl
from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

def visualizza_predizioni(knn_model, X, y):
    """
    Funzione ausiliaria per mostrare a video il confronto diretto
    tra l'etichetta REALE e quella PREDETTA dal KNN.
    Usa una strategia Holdout semplificata (70% train, 30% test).
    """
    print("\n------------------------------------------------------------")
    print("                 TEST VISIVO (Reale vs Predetto)")
    print("------------------------------------------------------------")

    # 1. Dividiamo manualmente i dati (senza usare sklearn per coerenza col progetto)
    # Prendiamo il 70% per training e il restante 30% per testare
    split_index = int(len(X) * 0.7)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    if len(X_test) == 0:
        print("Dataset troppo piccolo per dividere in train/test!")
        return

    # 2. Addestriamo il modello (memorizza i dati di train)
    knn_model.fit(X_train, y_train)

    # 3. Facciamo le predizioni sul test set
    predictions = knn_model.predict(X_test)

    # 4. Creiamo un DataFrame per visualizzare bene i risultati
    results = pd.DataFrame({
        'Reale': y_test.values,
        'Predetto': predictions
    })

    # Aggiungiamo una colonna per vedere se ha indovinato
    results['Esito'] = results.apply(
        lambda row: "CORRETTO" if row['Reale'] == row['Predetto'] else "ERRATO", axis=1
    )

    print(results)
    print("------------------------------------------------------------\n")

def main():
    print("--- FASE 1: Preprocessing ---")

    # 1. Definiamo il percorso del file
    # È importante usare os.path.join per compatibilità tra Windows/Mac/Linux
    dataset_path = os.path.join("dati", "toy_dataset_1.csv")

    # Verifica di sicurezza
    if not os.path.exists(dataset_path):
        print(f"Errore: Il file '{dataset_path}' non esiste.")
        return

    # 2. Istanziamo il Preprocessor
    # Usiamo PreprocessorImpl come definito nel tuo file
    preprocessor = PreprocessorImpl()

    try:
        # 3. Eseguiamo il preprocessing
        # Il metodo restituisce la tupla (X, y) già separate e pulite
        print(f"Elaborazione del file: {dataset_path} ...")
        X, y = preprocessor.preprocess(dataset_path)

        print("\n=== DATASET PREPROCESSATO COMPLETO ===")

        print("\n--- FEATURE MATRIX X ---")
        print(X.to_string(index=True))

        print("\n--- TARGET y ---")
        print(y.to_string(index=True))

        # 4. Verifica dei risultati (Feedback per l'utente)
        print("Preprocessing completato con successo!")
        print(f" -> Dimensioni Feature (X): {X.shape}")     # (Righe, Colonne)
        print(f" -> Dimensioni Target (y):  {y.shape[0]}")
        print(f" -> Feature estratte: {list(X.columns)}")   # Nomi delle colonne

    except Exception as e:
        print(f"Errore durante il preprocessing: {e}")

    print("\n --- FASE 2: KNNAlgorithm ---")

    # Chiediamo K all'utente
    try:
        k_input = input("    -> Inserisci K (numero vicini, invio per default=1): ")
        k = int(k_input) if k_input.strip() else 1
    except ValueError:
        print("       ! Valore non valido. Uso K=1.")
        k = 1

    # Istanziamo l'algoritmo KNN
    # Nota: Usiamo 'euclidian' come metrica standard definita nel tuo DistanceFactory
    knn_model = KnnAlgorithm(k=k, metric_name='euclidian')
    print(f"    -> Modello KNN creato (K={k}, Metrica='euclidian')")

    # Chiamiamo la funzione che abbiamo creato sopra
    visualizza_predizioni(knn_model, X, y)

if __name__ == "__main__":
    main()