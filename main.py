import os
import sys

# -------------------------------------------------------------------------
# IMPORTAZIONI
# -------------------------------------------------------------------------
try:
    from Preprocessing.Implementation.PreprocessorImpl import PreprocessorImpl
    from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm
    from EvaluationModel.Factory.EvaluationFactory import EvaluationFactory
except ImportError as e:
    print(f"ERRORE DI IMPORTAZIONE: {e}")
    print("Assicurati di eseguire questo script dalla cartella principale del progetto.")
    sys.exit(1)


def main():
    print("============================================================")
    print("   PROGETTO AI - CLASSIFICATORE KNN (GRUPPO 9)")
    print("============================================================")

    # ---------------------------------------------------------------------
    # 1. CARICAMENTO E PULIZIA DATI
    # ---------------------------------------------------------------------
    print("\n[1] Caricamento Dati...")

    # MODIFICA: Puntiamo al nuovo file (che devi aver salvato come CSV)
    nome_file = "version_1.csv"
    dataset_path = os.path.join("dati", nome_file)

    # Verifica esistenza file
    if not os.path.exists(dataset_path):
        print(f"ERRORE: File non trovato in '{dataset_path}'")
        print(f"NOTA: Assicurati di aver convertito il file .numbers in .csv e di averlo messo nella cartella 'dati'.")
        return

    # Istanzio il preprocessore e pulisco i dati
    preprocessor = PreprocessorImpl()
    try:
        # Nota: Il preprocessor rimuoverà automaticamente "Blood Pressure" e "Heart Rate"
        # presenti anche in questo nuovo dataset.
        X, y = preprocessor.preprocess(dataset_path)
        print(f"    -> Dataset caricato: {len(X)} campioni, {len(X.columns)} features.")
    except Exception as e:
        print(f"    -> Errore nel preprocessing: {e}")
        return

    # ---------------------------------------------------------------------
    # 2. CONFIGURAZIONE PARAMETRI
    # ---------------------------------------------------------------------
    print("\n[2] Configurazione...")

    try:
        k_input = input("    -> Inserisci K (numero di vicini, default=5): ")
        k = int(k_input) if k_input.strip() else 5

        print("    -> Scegli metodo di valutazione:")
        print("       1. Holdout (80% train - 20% test)")
        print("       2. K-Fold Cross Validation")
        print("       3. Leave-One-Out")
        method_input = input("    -> Scelta (1-3, default=1): ")

        method_map = {'1': 'holdout', '2': 'kfold', '3': 'loo'}
        method_name = method_map.get(method_input.strip(), 'holdout')

    except ValueError:
        print("    -> Valore non valido. Uso impostazioni di default (K=5, Holdout).")
        k = 5
        method_name = 'holdout'

    print(f"    -> Esecuzione con K={k} e Metodo='{method_name}'")

    # ---------------------------------------------------------------------
    # 3. ESECUZIONE ALGORITMO
    # ---------------------------------------------------------------------
    print("\n[3] Avvio Analisi...")

    try:
        # A. Creiamo il modello KNN
        knn_model = KnnAlgorithm(k=k, metric_name='euclidian')

        # B. Creiamo la strategia di valutazione
        evaluator = EvaluationFactory.create(method_name)

        # C. Eseguiamo la valutazione
        results = evaluator.evaluate(model=knn_model, X=X, y=y, k_neighbors=k)

    except Exception as e:
        print(f"\nERRORE CRITICO DURANTE L'ESECUZIONE:\n{e}")
        return

    # ---------------------------------------------------------------------
    # 4. STAMPA RISULTATI
    # ---------------------------------------------------------------------
    print("\n============================================================")
    print("                     RISULTATI FINALI")
    print("============================================================")

    metrics = results.get("mean", {})

    print(f"Metodo utilizzato: {results.get('method', 'N/A').upper()}")
    print("-" * 30)
    print(f"Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"Sensitivity: {metrics.get('sensitivity', 0):.4f}")
    print(f"Specificity: {metrics.get('specificity', 0):.4f}")
    print(f"G-Mean:      {metrics.get('gmean', 0):.4f}")
    print(f"AUC:         {metrics.get('auc', 0):.4f}")
    print("-" * 30)
    print("Analisi completata con successo.")


if __name__ == "__main__":
    main()