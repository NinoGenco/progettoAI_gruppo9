import os
import pandas as pd
from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl

def main():
    """
    Main di test per verificare il corretto funzionamento del preprocessing.
    Non esporta file: stampa tutto a terminale.
    """

    # Percorso al dataset
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, "dati", "version_1.csv")

    # Inizializzazione preprocessore
    preprocessor = PreprocessorImpl()

    # Esecuzione preprocessing
    X, y = preprocessor.preprocess(data_path)

    # STAMPE DI CONTROLLO
    print("\n=== PREPROCESSING COMPLETATO ===")

    print("\nPrime 5 righe delle FEATURES (X):")
    print(X.head())

    print("\nPrime 5 righe del TARGET (y):")
    print(y.head())

    print("\nDimensioni dataset:")
    print(f"Features (X): {X.shape}")
    print(f"Target   (y): {y.shape}")

    print("\n=== CONTROLLI DI COERENZA ===")
    print(f"Numero di campioni X == y ? {X.shape[0] == y.shape[0]}")
    print(f"Valori unici nel target: {sorted(y.unique())}")
    print(f"Valori NaN in X: {X.isna().sum().sum()}")
    print(f"Valori NaN in y: {y.isna().sum()}")

    # STAMPA COMPLETA DATASET
    # ATTENZIONE: stampa ~700 righe

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\n=== DATASET PREPROCESSATO COMPLETO (X) ===")
    print(X)

    print("\n=== TARGET COMPLETO (y) ===")
    print(y)


if __name__ == "__main__":
    main()