from typing import Tuple
import pandas as pd
import numpy as np

from Preprocessing.Preprocessor import Preprocessor

class PreprocessorImpl(Preprocessor):
    """
    Implementazione del Preprocessor conforme alla traccia del corso.

    Dataset: Wisconsin Breast Cancer (Original)

    Operazioni eseguite:
    - caricamento del file CSV
    - rimozione dell'ID (Sample code number)
    - rimozione di colonne NON previste (Blood Pressure, Heart Rate)
    - rinomina delle colonne non standard
    - gestione di valori non numerici ('?', stringhe)
    - gestione numeri europei (',' → '.')
    - imputazione dei valori mancanti SOLO sulle feature (mediana)
    - mantenimento del dominio delle feature [1, 10]
    - forzatura dell'ordine ufficiale delle feature
    - separazione in feature (X) e target (y)

    NOTE IMPORTANTI (come da traccia):
    - nessuna normalizzazione
    - nessuna standardizzazione
    - target NON modificato (rimane 2 / 4)
    - ordine delle righe invariato
    """

    def __init__(self):

        # Nome della colonna target
        self.target_column = "classtype_v1"

        # Colonne da eliminare (ID + non previste dalla traccia)
        self.columns_to_drop = [
            "Sample code number",
            "Blood Pressure",
            "Heart Rate"
        ]

        # Rinomina colonne non standard
        self.columns_rename = {
            "uniformity_cellsize_xx": "Uniformity of Cell Size",
            "clump_thickness_ty": "Clump Thickness",
            "bareNucleix_wrong": "Bare Nuclei"
        }

        # Ordine ufficiale delle feature (TRACCIA DEL CORSO)
        self.ordered_features = [
            "Clump Thickness",
            "Uniformity of Cell Size",
            "Uniformity of Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses"
        ]

    def preprocess(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:

        """ Esegue l'intero preprocessing del dataset.

        :param data_path: percorso al file CSV
        :return:
            X → DataFrame delle feature
            y → Series del target (Class: 2 / 4) """

        # 1. Caricamento dataset
        df = pd.read_csv(data_path)

        # Rimozione spazi dai nomi delle colonne
        df.columns = df.columns.str.strip()

        # 2. Rinomina colonne non standard
        df = df.rename(columns=self.columns_rename)

        # 3. Rimozione colonne non richieste
        df = df.drop(
            columns=[col for col in self.columns_to_drop if col in df.columns]
        )

        # 4. Pulizia valori non numerici
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace("?", np.nan)
                )

        # 5. Conversione a numerico
        df = df.apply(pd.to_numeric, errors="coerce")

        # 6. Eliminazione righe con target mancante
        df = df.dropna(subset=[self.target_column])

        # 7. Imputazione valori mancanti sulle FEATURE
        for col in self.ordered_features:
            df[col] = df[col].fillna(df[col].median())

        # 8. Vincolo dominio feature [1, 10]
        for col in self.ordered_features:
            df[col] = (
                df[col]
                .round()
                .clip(1, 10)
                .astype(int)
            )

        # 9. Separazione X e y + ordine colonne
        X = df[self.ordered_features]
        y = df[self.target_column]

        return X, y