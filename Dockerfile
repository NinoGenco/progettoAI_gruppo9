# 1. Usa un'immagine base di Python (versione 3.10 leggera per risparmiare spazio)
FROM python:3.12-slim

# 2. Imposta la cartella di lavoro dentro il container a "/app"
WORKDIR /app

# 3. Installa strumenti di sistema di base (utili per compilare librerie come Numpy/Pandas se necessario)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia il file requirements.txt dal tuo computer al container
COPY requirements.txt .

# 5. Installa le librerie Python elencate nel file requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia tutto il resto del codice sorgente dal tuo computer alla cartella /app del container
COPY . .

# 7. Crea la cartella 'plots' dentro il container (per assicurarsi che esista prima di scriverci)
RUN mkdir -p plots

# 8. Comando di avvio: quando il container parte, esegue il file main.py
CMD ["python", "main.py"]