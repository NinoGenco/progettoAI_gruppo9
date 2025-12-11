import pandas as pd
import numpy as np
from typing import Tuple
from pandas import DataFrame, Series
from Preprocessing.Preprocessor import Preprocessor

""" Questa classe implementa l’interfaccia Preprocessor e si occupa di:
   1) Caricare il dataset dal file CSV.
   2) Rimuovere le colonne di non interesse secondo la traccia.
   3) Correggere numeri reali mettendo i decimali dopo il punto.
   4) Eliminare righe duplicate.
   5) Gestire valori mancanti o anomali.
   6) Convertire i dati sottoforma di stringhe in formato numerico.
   7) Inserire i valori mancanti usando la media.
   8) Dividere il dataset in features (X) e target (y)."""

#1) Caricare il dataset dal file CSV.

class SimpleCSVLoader(Preprocessor):

    def __init__(self):
        pass

    def preprocess(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        #Carichiamo il file CSV.
        df = pd.read_csv(data_path)

        #X diventa l'intero dataset.
        X = df

        #y diventa un DataFrame vuoto (per rispettare l'interfaccia che vuole 2 output)
        y = pd.DataFrame()

        return X, y
