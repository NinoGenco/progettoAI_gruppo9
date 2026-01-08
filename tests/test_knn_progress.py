import sys
import os
import numpy as np

# Questo blocco serve a far capire a Python dove trovare la cartella KNNAlgorithm
# indipendentemente da dove posizioni questo file.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Se questo file è nella root, project_root è current_dir.
# Se è in una sottocartella, risaliamo finché non troviamo la root.
project_root = current_dir
if os.path.basename(project_root) == 'KNNAlgorithm':
    project_root = os.path.dirname(project_root)

sys.path.append(project_root)

from KNNAlgorithm.KnnAlgorithm import KnnAlgorithm

def test_knn_complete_flow():
    print("--- INIZIO TEST KNN (MANUALE) ---")

    # CREIAMO DATI DI ESEMPIO
    # Immagina due gruppi di punti su un piano cartesiano:
    # Gruppo A (Label 0): Punti vicini a (1, 1)
    # Gruppo B (Label 1): Punti vicini a (5, 5)

    # Usiamo liste Python standard (l'algoritmo le convertirà in numpy array)
    X_train = [
        [1, 1], [1, 2], [2, 1],  # Gruppo A (vicini all'origine)
        [5, 5], [5, 6], [6, 5]  # Gruppo B (lontani dall'origine)
    ]

    y_train = [0, 0, 0, 1, 1, 1]  # Le etichette corrispondenti (0=A, 1=B)

    print(f"Dati di Training creati: {len(X_train)} esempi.")

    # ISTANZIAMO L'ALGORITMO
    # Qui stiamo testando la FACTORY: passando 'euclidian', la factory deve
    # restituire l'oggetto SquaredEuclidianDistanceStrategy.
    k_value = 3
    metric = 'euclidian'

    print(f"Inizializzazione KNN con k={k_value} e metrica='{metric}'...")

    try:
        knn = KnnAlgorithm(k=k_value, metric_name=metric)
    except Exception as e:
        print(f"Errore inizializzazione: {e}")
        return

    # ADDESTRAMENTO
    knn.fit(X_train, y_train)
    print("Addestramento completato.")

    # PREDIZIONE (TESTING)
    # Creiamo due punti nuovi di test:
    # Punto 1: (1.5, 1.5) -> Dovrebbe essere CLASSE 0 (è vicino al gruppo A)
    # Punto 2: (5.5, 5.5) -> Dovrebbe essere CLASSE 1 (è vicino al gruppo B)
    X_test = [
        [1.5, 1.5],
        [5.5, 5.5]
    ]

    print(f"Avvio predizione su: {X_test}...")

    # Qui stiamo testando la STRATEGY: il calcolo delle distanze avviene ora
    predictions = knn.predict(X_test)

    # VERIFICO RISULTATI
    print(f"Predizioni ottenute: {predictions}")

    expected_results = [0, 1]

    if np.array_equal(predictions, expected_results):
        print("\n TEST SUPERATO! Il sistema funziona correttamente.")
        print(" Il punto (1.5, 1.5) è stato classificato come 0.")
        print(" Il punto (5.5, 5.5) è stato classificato come 1.")
    else:
        print("\n TEST FALLITO. Qualcosa non va nei calcoli.")
        print(f" Atteso: {expected_results}")
        print(f" Ottenuto: {predictions}")

if __name__ == "__main__":
    try:
        test_knn_complete_flow()
    except Exception as e:
        print(f"\n ERRORE CRITICO DURANTE L'ESECUZIONE:\n{e}")