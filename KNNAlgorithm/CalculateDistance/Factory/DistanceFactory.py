from abc import ABC
from KNNAlgorithm.CalculateDistance.Strategy.EuclidianDistance import EuclideanDistance

class DistanceFactory(ABC):

    """ Questa classe implementa il Design Pattern 'Factory Method' per la creazione delle strategie di calcolo della distanza.
    Serve a gestire la scelta dell'algoritmo in un unico punto del codice, semplificando il lavoro per il resto del
    programma, il quale deve solo chiedere la metrica desiderata.

    La classe non possiede variabili o dati interni da memorizzare"""

    @staticmethod
    def get_distance_metric(metric_name: str):

        """Riceve il nome della metrica come testo e restituisce l'oggetto corrispondente pronto all'uso. Pulisce la
        stringa ricevuta per evitare errori se l'utente scrive il nome in modo impreciso.

        Parametri: metric_name: La stringa che indica quale formula di distanza si vuole usare.

        Risultati: Restituisce l'oggetto che calcola la distanza richiesta. Se il nome non è valido segnala un errore."""

        # Pulisce il testo in input, in particolare rimuove spazi e converte tutto in minuscolo.
        clean_name = metric_name.lower().strip()

        if clean_name == 'euclidian' or clean_name == 'euclidean':
            return EuclideanDistance()
        else:
            raise ValueError(f"La metrica '{metric_name}' non è supportata o non esiste.")