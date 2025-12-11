#1) Caricare il dataset dal file CSV.
from Preprocessing.Implementation.PreprocessorImpl import SimpleCSVLoader

# Creo l'istanza
loader = SimpleCSVLoader()

#Carico tutti i dati del dataset sulla variabile X
X = loader.preprocess("Dati/version_1.csv")

#Verifico.
print(X)