from typing import Tuple
import pandas as pd
import numpy as np

from Preprocessing.Preprocessor import Preprocessor

class PreprocessorImpl(Preprocessor):

    """ Implementazione del Preprocessor.
        Interfaccia astratta per il preprocessing del dataset.

        Seguendo questo approccio, è possibile cambiare la logica di preprocessing senza modificare il codice che utilizza il Preprocessor.

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

    NOTE IMPORTANTI:
    - nessuna normalizzazione
    - nessuna standardizzazione
    - target NON modificato (rimane 2 / 4)
    - ordine delle righe invariato """

    def __init__(self):

        # Nome della colonna target
        self.target_column = "classtype_v1"

        # Colonne da eliminare (ID + non previste dalla traccia)
        self.columns_to_drop = [
            "Sample code number",  # Identificativo (non predittivo)
            "Blood Pressure",      # Colonna non prevista
            "Heart Rate"           # Colonna non prevista
        ]

        # Rinomina colonne non standard
        self.columns_rename = {
            "uniformity_cellsize_xx": "Uniformity of Cell Size",
            "clump_thickness_ty": "Clump Thickness",
            "bareNucleix_wrong": "Bare Nuclei"
        }

        # Ordine ufficiale delle feature (TRACCIA DEL CORSO), serve a garantire coerenza nell'input del modello
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

        # 3. Rimozione colonne non richieste solo se effettivamente presenti
        df = df.drop(
            columns=[col for col in self.columns_to_drop if col in df.columns]
        )

        # 4. Pulizia valori non numerici; sostituisco virgola con punto e converte '?' in NaN
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)    #Forza la conversione a stringa
                    .str.replace(",", ".", regex=False)     #Normalizza formato decimale
                    .replace("?", np.nan)       #Sostituisce placeholder con NaN
                )

        # 5. Conversione a numerico, gli errori residui vengono trasformati in NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        # 6. Eliminazione righe con target mancante
        df = df.dropna(subset=[self.target_column])

        # 7. Imputazione valori mancanti SOLO sulle FEATURE, utilizzo la MEDIANA
        for col in self.ordered_features:
            df[col] = df[col].fillna(df[col].median())

        # 8. Vincolo dominio feature [1, 10]
        # - Arrotonda i valori, limita intervallo, converte a intero mantenendo la natura originale del dataset
        for col in self.ordered_features:
            df[col] = (
                df[col]
                .round()    #Arrotondamento
                .clip(1, 10)    #Vincolo dominio
                .astype(int)    #Conversione a intero
            )

        # 9. Separazione tra feature (X) e target (y) + ordine colonne
        X = df[self.ordered_features]
        y = df[self.target_column]

        return X, y #Restitutisce il dataset pronto per il classificatore
