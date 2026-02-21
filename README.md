# Progetto AI per la classificazione di tumori

## INDICE
1. [DESCRIZIONE PROGETTO]
2. [PRE-PROCESSING]
3. [ALGORITMO K-NN]
4. [EVALUATION]
5. [METRICHE DI VALUTAZIONE]
6. [RISULTATI]
7. [ISTRUZIONI DOCKER]


### 1. DESCRIZIONE  PROGETTO

Il programma addestra e valuta le prestazioni di un classificatore di Machine Learning basato sull'algoritmo K-NN.
L'obiettivo applicativo è la classificazione binaria di tumori benigni o maligni, a partire dalla visione di alcuni dati
medici. Il sistema è progettato per supportare il processo decisionale fornendo non solo una predizione, ma anche 
delle metriche sull'affidabilità del modello. Si parte dalla pulizia del dataset, fornito tramite file CSV, passando alla
classificazione tramite distanza Euclidea, fino alla validazione statistica (Holdout, K-Fold, Leave-One-Out). Il sistema
fornisce un'analisi completa delle performance salvando automaticamente i risultati, tramite file CSV, ed i grafici delle
matrici di confusione.

### 2. PRE-PROCESSING

Questa fase permette di trasformare i dati grezzi in un formato più pulito, strutturato ed adatto all'analisi. Il processo
si sviluppa in diverse fasi:


- Caricamento Dataset, vengono letti i dati da un file di input in formato CSV.


- Eliminazione delle colonne non necessarie, vengono rimosse Feature irrilevanti per l'analisi, e rinominate quelle con
intestazioni errate.


- Conversione dei dati, i valori vengono trasformati in un formato numerico gestendo le differenze di formattazione, ad
esempio conversione della virgola in punto decimale. Eventuali errori di conversione possono portare alla rimozione delle
righe problematiche.


- Gestione dei valori mancanti, le righe in cui è mancante la variabile Target vengono rimosse, mentre i valori mancanti 
nelle Feature vengono imputati, sostituendoli con la mediana della colonna stessa. Questa decisione preserva la quantità
di dati.


- Normalizzazione, non viene applicata una standardizzazione classica, ma i dati vengono forzati all'interno dell'intervallo
[1,10].


- Separazione di Features e Target, consiste nella suddivisione del dataset in due insiemi distinti, Feature (X)
contenente la matrice dei dati usata per la predizione, e Target (Y) che rappresenta la variabile di output che il modello
dovrà predire.

### 3. ALGORITMO K-NN

L'algoritmo dei k-nearest neighbors (KNN) è un metodo di classificazione supervisionata, che assegna ad un nuovo dato la
classe più frequente tra i suoi k vicini più prossimi. Le caratteristiche principali sono:

- Ricerca dei vicini più prossimi, per ogni punto di test viene calcolata la distanza rispetto a tutti i punti del dataset
di training, utilizzando la strategia di distanza selezionata. Vengono cos' selezionati i k elementi più vicini.


- Scelta della strategia di distanza, il sistema utilizza un'architettura basata sui pattern Factory e Strategy,
consentendo di selezionare dinamicamente il metodo di calcolo della distanza. Questo approccio garantisce una maggiore
flessibilità, facilitando l'integrazione di nuove strategie di distanza senza modificare il codice esistente, come la
distanza di Manhattan o la distanza di Minkowski. Attualmente è implementata solamente la distanza euclidea.


- Classificazione, una volta individuati i k vicini più prossimi, la classe viene assegnata in base alla maggioranza tra
le etichette dei vicini. In caso di parità tra classi, il programma sceglie casualmente una delle classi con il numero
massimo di occorrenze.

Il parametro "k" rappresenta il numero di vicini da prendere in considerazione per il processo di classificazione. Se si
imposta k=1, l'oggetto da classificare viene associato alla categoria del punto più vicino, rischiando Overfitting.
La scelta ottimale di "k" è cruciale per bilanciare correttamente l'accuratezza del modello e la sua capacità di
generalizzazione. Il software include controlli di validazione per assicurare che i parametri inseriti siano corretti,
restituendo messaggi di errore in caso di configurazioni non valide.

### 4. EVALUATION

Per valutare le prestazioni del classificatore, il programma implementa tre tecniche di validazione:

- Holdout Validation. In questa metodologia il dataset viene suddiviso in due insiemi separati, uno di addestramento
(Training Set), usato dal modello per apprendere, ed uno di test (Test Set), usato per valutare il modello. La
percentuale di dati destinata all'addestramento ed al test è determinata dall'utente. Il modello, una volta addestrato,
prova a indovinare la classe dei pazienti nel Test Set. Confrontando le sue previsioni con la realtà, il sistema calcola
le metriche di affidabilità per comprendere le prestazioni del modello.


- K-Fold Cross Validation. Il dataset viene suddiviso in k sottoinsiemi, chiamati folds, di uguali dimensioni o simili,
dopodiché il modello viene addestrato su k-1 di questi sottoinsiemi e testato sull’ultimo rimanente. Tale processo viene
iterato per k volte, cambiando il fold di test ogni volta. Utilizziamo una suddivisione Stratificata, in modo tale da 
assicurare che ogni fold mantenga la stessa proporzione di classi Benigno/Maligno del dataset originale, evitando
sbilanciamenti che potrebbero falsare il training. Alla fine, le metriche di valutazione vengono mediate su tutte le
iterazioni, in modo da ottenere una stima più affidabile delle prestazioni. Questo metodo è utile per ridurre il rischio
di overfitting rispetto alla semplice suddivisione tra training e test set, poiché il modello viene trainato e testato
su diverse parti del dataset.


- Leave-One-Out.  In questa strategia, il numero di folds è esattamente uguale al numero totale di campioni presenti nel
dataset.Per ogni iterazione, il sistema isola un singolo campione come Test Set, mentre i dati rimanenti vengono
 utilizzati come Training Set. Questo processo viene ripetuto per ogni riga del dataset.

### 5. METRICHE DI VALUTAZIONE

- Accuracy Rate: L'accuratezza misura la percentuale di previsioni corrette rispetto al totale delle previsioni effettuate.


- Error Rate: Il tasso di errore misura la percentuale di previsioni errate rispetto al totale delle previsioni effettuate.


- Sensitivity: La sensibilità misura la capacità del modello di identificare correttamente i positivi, ossia quanti veri
positivi sono stati correttamente identificati.


- Specificity: La specificità misura la capacità del modello di identificare correttamente i negativi, ossia quanti veri
negativi sono stati correttamente identificati.


- Geometric Mean: La media geometrica è una metrica che combina la sensibilità e la specificità in un unico valore,
utilizzato per avere un indicatore equilibrato delle prestazioni del modello.

### 6. RISULTATI

Dopo l'esecuzione della valutazione, il programma produce due output principali. Il primo è un file CSV che contiene i
valori delle metriche selezionate, calcolati in base alle predizioni effettuate dal modello. Questo file permette di
analizzare le prestazioni del modello in modo dettagliato e quantitativo. Il secondo output è un plot della matrice di
confusione, salvato come immagine, che fornisce una rappresentazione visiva degli errori e delle corrette classificazioni
effettuate. Questo grafico aiuta a comprendere meglio il comportamento del modello, specialmente in presenza di classi
sbilanciate.

### 7. ISTRUZIONI DOCKER

Il progetto è configurato per essere eseguito all'interno di un container Docker, garantendo un ambiente isolato e riproducibile.

#### Requisiti
- Docker installato sulla macchina.

#### 1. Costruire l'immagine
Aprire il terminale nella cartella radice del progetto ed eseguire:

    docker build -t progetto_ai_gruppo9 .

#### 2. Eseguire il container
Il comando seguente avvia il container mappando le cartelle locali per permettere al programma di leggere il dataset e salvare i risultati (grafici e report) direttamente sul tuo computer.

Nota: Prima di eseguire, assicurarsi che esista un file (anche vuoto) chiamato `report_performance.csv` nella cartella del progetto per evitare che Docker lo crei come una directory.

**Su Linux / Mac:**
    touch report_performance.csv
    docker run -it --rm \
      -v "$(pwd)/dati":/app/dati \
      -v "$(pwd)/plots":/app/plots \
      -v "$(pwd)/report_performance.csv":/app/report_performance.csv \
      progetto_ai_gruppo9

**Su Windows (PowerShell):**
    if (-not (Test-Path .\report_performance.csv)) { New-Item .\report_performance.csv -ItemType File }
    docker run -it --rm `
      -v ${PWD}\dati:/app/dati `
      -v ${PWD}\plots:/app/plots `
      -v ${PWD}\report_performance.csv:/app/report_performance.csv `
      progettoAI_gruppo9

##### Spiegazione dei volumi:
- `-v .../dati:/app/dati`: Passa il dataset locale al container.
- `-v .../plots:/app/plots`: Salva i grafici generati nella cartella 'plots' del tuo computer.
- `-v .../report_performance.csv:/app/report_performance.csv`: Salva/Aggiorna il file CSV dei risultati sul tuo computer.