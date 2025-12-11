import pandas as pd
import numpy as np
from Preprocessing.Preprocessor import Preprocessor

""" Questa classe implementa l’interfaccia Preprocessor e si occupa di effettuare 
    tutte le operazioni richieste dal progetto:
    
   - pulizia colonne
   - rimozione duplicati
   - gestione dei valori mancanti
   - conversione numeri "europei"
   - rimozione ID
   - separazione X e y. """

class PreprocessorImpl(Preprocessor):

    def preprocess(self, data_path: str):

        df = pd.read_csv(data_path, dtype=str)
        # dtype=str → necessario per gestire valori europei tipo "3,14"

        first_col = df.columns[0].lower()
        if "id" in first_col or "sample" in first_col:
            df = df.drop(columns=[df.columns[0]])

        df = df.drop_duplicates()

        df = df.apply(lambda col: col.str.replace(",", ".", regex=False))

        df = df.replace(["?", "", " ", "NaN", "nan"], np.nan)

        df = df.astype(float)

        df = df.fillna(df.mean())

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        return X, y