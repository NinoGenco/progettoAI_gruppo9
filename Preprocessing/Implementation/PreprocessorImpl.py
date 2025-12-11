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

        # 2. RIMOZIONE DELLA COLONNA ID

        first_col = df.columns[0].lower()
        if "id" in first_col or "sample" in first_col:
            df = df.drop(columns=[df.columns[0]])

        # 3. RIMOZIONE DUPLICATI

        df = df.drop_duplicates()

        # 4. NORMALIZZAZIONE NUMERI EUROPEI "12,5" → "12.5"

        df = df.apply(lambda col: col.str.replace(",", ".", regex=False))

        # 5. GESTIONE VALORI MANCANTI

        df = df.replace(["?", "", " ", "NaN", "nan"], np.nan)

        # 6. CONVERSIONE A NUMERICO

        df = df.astype(float)

        # 7. IMPUTAZIONE DEI VALORI MANCANTI

        df = df.fillna(df.mean())

        # 8. DIVISIONE IN FEATURE E TARGET

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        return X, y