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

