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

    @abstractmethod
    def preprocess(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        pass