import os
import sys

# Assicuriamoci che Python trovi i moduli (opzionale ma consigliato)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiamo l'implementazione concreta
from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl

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

        # 4. Verifica dei risultati (Feedback per l'utente)
        print("Preprocessing completato con successo!")
        print(f" -> Dimensioni Feature (X): {X.shape}")     # (Righe, Colonne)
        print(f" -> Dimensioni Target (y):  {y.shape[0]}")
        print(f" -> Feature estratte: {list(X.columns)}")   # Nomi delle colonne

    except Exception as e:
        print(f"Errore durante il preprocessing: {e}")

if __name__ == "__main__":
    main()