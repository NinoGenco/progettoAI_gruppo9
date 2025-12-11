import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from Preprocessing.Preprocessor import Preprocessor


""" Questa classe implementa l’interfaccia Preprocessor e si occupa di:
   - caricare il dataset dal file CSV
   - rimuovere colonne non utili (come ID)
   - eliminare righe duplicate
   - gestire valori mancanti o anomali ("?", stringhe vuote, ecc.)
   - correggere numeri in formato europeo (es. "3,14" → "3.14")
   - convertire tutto in formato numerico
   - imputare i valori mancanti usando la media
   - dividere il dataset in features (X) e target (y) """

class PreprocessorImpl(Preprocessor):

    # Nome della colonna target del dataset del progetto
    __class_column_name = "Class"

    # 1. METODO PRINCIPALE richiesto dall’interfaccia

    def preprocess(self, data_path: str) -> tuple[DataFrame, Series]:
        # Caricamento da CSV (come stringhe per gestire numeri europei)
        df = self.load_dataset(data_path)

        # Pulizia delle colonne e delle righe
        df = self.data_cleanup(df)

        # Standardizzazione opzionale dei dati (se richiesta dalla pipeline)
        df = self.data_standardization(df)

        # Divisione in features (X) e target (y)
        X, y = self.get_targets_and_features(df)

        return X, y

    # CARICAMENTO DEL DATASET

    def load_dataset(self, data_path: str) -> pd.DataFrame:
        # Carichiamo tutto come stringa per evitare errori di parsing
        df = pd.read_csv(data_path, dtype=str)
        return df

    # PULIZIA DATI E COLONNE

    def data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:

        # Rimuove la colonna ID o simili se presente come prima colonna
        first_col = df.columns[0].lower()
        if "id" in first_col or "sample" in first_col:
            df = df.drop(columns=[df.columns[0]])

        # Rimozione righe duplicate
        df = df.drop_duplicates()

        # Conversione dei numeri europei ("3,14") in formato standard ("3.14")
        df = df.apply(lambda col: col.str.replace(",", ".", regex=False))

        # Sostituzione valori mancanti o anomali con NaN
        df = df.replace(["?", "", " ", "NaN", "nan"], np.nan)

        # Conversione di ogni colonna in float
        df = df.astype(float)

        # Imputazione dei NaN con la media della colonna
        df = df.fillna(df.mean())

        return df

    # STANDARDIZZAZIONE DEI DATI (OPZIONALE)

    def data_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        # Memorizziamo la colonna target in variabile temporanea
        y = df[self.__class_column_name]

        # Rimuoviamo temporaneamente la colonna del target
        df_no_class = df.drop(columns=[self.__class_column_name])

        # Standardizzazione: (valore - media) / deviazione standard
        df_standardized = (df_no_class - df_no_class.mean()) / df_no_class.std()

        # Aggiungiamo nuovamente la colonna target
        df_standardized[self.__class_column_name] = y

        return df_standardized

    # SEPARAZIONE FEATURES (X) E TARGET (y)

    def get_targets_and_features(self, df: pd.DataFrame) -> tuple[DataFrame, Series]:

        # La colonna target è sempre l’ultima del dataset
        y = df.iloc[:, -1]      # variabile dipendente (Class)
        X = df.iloc[:, :-1]     # variabili indipendenti (features)

        return X, y