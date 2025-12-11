from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

""" Questa interfaccia definisce il comportamento che ogni classe di preprocessing deve avere.
È stata progettata per fornire flessibilità: diverse classi possono implementare logiche
di preprocessing differenti, purché espongano un metodo `preprocess` con le caratteristiche
descritte di seguito.

Lo scopo è permettere al progetto di cambiare facilmente il modo in cui i dati vengono
preprocessati senza modificare il resto della pipeline. """

class Preprocessor(ABC):

    """ Metodo astratto che deve essere implementato da qualunque classe in grado di
        preprocessare un dataset.

        Il metodo riceve:
            - il percorso al file contenente il dataset

        e deve restituire:
            - una tupla (X, y)
                X → DataFrame contenente le feature
                y → DataFrame o Serie contenente la variabile target

        Ogni classe concreta può implementare il preprocessing in modi diversi
        (rimozione colonne, gestione valori mancanti, scaling, ecc.),
        mantenendo però la stessa interfaccia. """

    @abstractmethod
    def preprocess(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """ Preprocessa il dataset situato nel percorso specificato.

                :param data_path: Percorso al file del dataset da caricare e preprocessare.
                :return: Una tupla (X, y) in cui:
                         X → DataFrame con le feature preprocessate
                         y → DataFrame o Serie contenente la colonna target """
        pass