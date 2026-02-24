from abc import ABC
from KNNAlgorithm.CalculateDistance.Strategy.EuclideanDistance import EuclideanDistance

class DistanceFactory(ABC):

    """ Questa classe implementa il Design Pattern 'Factory Method' per la creazione delle strategie di calcolo della distanza.
    Serve a gestire la scelta dell'algoritmo in un unico punto del codice, semplificando il lavoro per il resto del
    programma, il quale deve solo chiedere la metrica desiderata. La classe non possiede variabili o dati interni da memorizzare"""

    @staticmethod
    def get_distance_metric(metric_name: str):

        """Riceve il nome della metrica come testo e restituisce l'oggetto corrispondente pronto all'uso.

        Parametri:
                  metric_name -> Stringa che indica quale formula di distanza si vuole usare.

        Risultati -> Restituisce l'oggetto che calcola la distanza richiesta."""

        # Converte in minuscolo il testo della stringa in input e rimuove spazi vuoti iniziali o finali.
        clean_name = metric_name.lower().strip()

        # Verifico se il nome corrisponde alla stringa 'euclidean'.
        if clean_name == 'euclidean':
            return EuclideanDistance()
        else:
            # Se la condizione if non è verificata blocco l'esecuzione del programma stampando un messaggio di errore.
            raise ValueError(f"La metrica '{metric_name}' non è supportata o non esiste.")