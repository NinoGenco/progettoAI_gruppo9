#1) Caricare il dataset dal file CSV.
from Preprocessing.Implementation.PreprocessorImpl import SimpleCSVLoader

# Creo l'istanza
loader = SimpleCSVLoader()

#Carico tutti i dati del dataset sulla variabile X
X = loader.preprocess("Dati/version_1.csv")

#2)Rimuovere colonne non di interesse secondo la traccia.
from Preprocessing.Implementation.PreprocessorImpl import ColumnDropper

# Lista delle colonne inutili.
colonne_inutili = ["Blood Pressure", "Heart Rate"]

# Inizializzo la classe passando la lista.
cleaner = ColumnDropper(columns_to_remove=colonne_inutili)

dataset_pulito, _ = cleaner.preprocess("Dati/version_1.csv")

# Verifico.
print("Colonne rimaste:", dataset_pulito.columns)# Lista delle colonne che sai essere inutili
print(dataset_pulito.head()) # Stampa le prime righe del tuo dataset completo
