import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from Preprocessing.Preprocessor import Preprocessor


class PreprocessorImpl(Preprocessor):

    # Nome della colonna target identificata nel dataset
    __class_column_name = "classtype_v1"

    # ======================================================
    # METODO PRINCIPALE richiesto dall'interfaccia
    # ======================================================
    def preprocess(self, data_path: str) -> tuple[DataFrame, Series]:

        # Caricamento del dataset
        df = self.load_dataset(data_path)

        # Pulizia dei dati e imputazione corretta
        df = self.data_cleanup(df)

        # Standardizzazione delle feature (non del target)
        df = self.data_standardization(df)

        # Divisione in feature (X) e target (y)
        X, y = self.get_targets_and_features(df)

        return X, y

    # ======================================================
    # 1. CARICAMENTO DEL DATASET
    # ======================================================
    def load_dataset(self, data_path: str) -> pd.DataFrame:
        # Legge tutto come stringa per evitare problemi con numeri in formato europeo
        df = pd.read_csv(data_path, dtype=str)
        return df

    # ======================================================
    # 2. PULIZIA DEI DATI — FA TUTTE LE OPERAZIONI RICHIESTE
    # ======================================================
    def data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:

        # ------------------------------
        # 🔹 Rimozione della colonna ID, se presente
        # ------------------------------
        id_candidates = ["Sample code number", "ID", "id", "sample", "sample_code"]

        for col in df.columns:
            if col in id_candidates or col.lower().startswith("sample"):
                df = df.drop(columns=[col])
                break

        # ------------------------------
        # 🔹 Rimozione righe duplicate
        # ------------------------------
        df = df.drop_duplicates()

        # ------------------------------
        # 🔹 Conversione numeri europei "3,14" → "3.14"
        # ------------------------------
        df = df.apply(lambda col: col.str.replace(",", ".", regex=False))

        # ------------------------------
        # 🔹 Gestione valori mancanti e anomali
        # ------------------------------
        df = df.replace(["?", "", " ", "NaN", "nan", "NULL"], np.nan)

        # ------------------------------
        # 🔹 Conversione di tutte le colonne in float
        # ------------------------------
        df = df.astype(float)

        # ------------------------------
        # 🔹 IMPUTAZIONE VALORI MANCANTI
        # ------------------------------

        # 1️⃣ Imputazione della TARGET con la MODA (valore più frequente)
        target_mode = df[self.__class_column_name].mode()[0]
        df[self.__class_column_name] = df[self.__class_column_name].fillna(target_mode)

        # 2️⃣ Imputazione delle FEATURE con la MEDIA
        feature_cols = df.columns[df.columns != self.__class_column_name]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

        return df

    # ======================================================
    # 3. STANDARDIZZAZIONE DELLE FEATURE
    # ======================================================
    def data_standardization(self, df: pd.DataFrame) -> pd.DataFrame:

        # Isola il target per NON scalarlo
        y = df[self.__class_column_name]

        # Rimuove temporaneamente la colonna target
        df_features = df.drop(columns=[self.__class_column_name])

        # Standardizzazione: (valore - media) / std
        df_scaled = (df_features - df_features.mean()) / df_features.std()

        # Riaggiunge la colonna target senza modifica
        df_scaled[self.__class_column_name] = y

        return df_scaled

    # ======================================================
    # 4. DIVISIONE FEATURE/TARGET
    # ======================================================
    def get_targets_and_features(self, df: pd.DataFrame) -> tuple[DataFrame, Series]:

        # Target label
        y = df[self.__class_column_name]

        # Tutte le altre colonne sono feature
        X = df.drop(columns=[self.__class_column_name])

        return X, y