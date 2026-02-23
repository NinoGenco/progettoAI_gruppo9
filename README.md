# Progetto AI per la classificazione di tumori


## INDICE
1. [INTRODUZIONE](#1-introduzione)
2. [DESCRIZIONE PROGETTO](#2-descrizione--progetto)
3. [PRE-PROCESSING](#3-pre-processing)
4. [ALGORITMO K-NN](#4-algoritmo-k-nn)
5. [EVALUATION](#5-evaluation)
6. [METRICHE DI VALUTAZIONE](#6-validazione-del-modello-e-metriche-calcolate)
7. [RISULTATI](#7-risultati)
8. [ESECUZIONE DEL PROGRAMMA](#8-esecuzione del programma)
9. [ISTRUZIONI DOCKER](#9-istruzioni-docker)
10. [CONCLUSIONE](#10-conclusione)


### 1. INTRODUZIONE

Il progetto è stato sviluppato da Antonino Genco, Andrea Nocera e Alberto Panocchi per il corso di Fondamenti di Intelligenza Artificiale (2025-2026).  


### 2. DESCRIZIONE PROGETTO

Il programma addestra e valuta le prestazioni di un classificatore di Machine Learning basato sull'algoritmo K-NN.
L'obiettivo applicativo è la classificazione binaria di tumori benigni o maligni, a partire dalla visione di alcuni dati
medici. Il sistema è progettato per supportare il processo decisionale fornendo non solo una predizione, ma anche 
delle metriche sull'affidabilità del modello. Si parte dalla pulizia del dataset, fornito tramite file CSV, passando alla
classificazione tramite distanza Euclidea, fino alla validazione statistica (Holdout, K-Fold, Leave-One-Out). Il sistema
fornisce un'analisi completa delle performance salvando automaticamente i risultati, tramite file CSV, ed i grafici delle
matrici di confusione.


### 3. PRE-PROCESSING

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


### 4. ALGORITMO K-NN

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


### 5. EVALUATION

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


### 6. METRICHE DI VALUTAZIONE

- Accuracy Rate: L'accuratezza misura la percentuale di previsioni corrette rispetto al totale delle previsioni effettuate.


- Error Rate: Il tasso di errore misura la percentuale di previsioni errate rispetto al totale delle previsioni effettuate.


- Sensitivity: La sensibilità misura la capacità del modello di identificare correttamente i positivi, ossia quanti veri
positivi sono stati correttamente identificati.


- Specificity: La specificità misura la capacità del modello di identificare correttamente i negativi, ossia quanti veri
negativi sono stati correttamente identificati.


- Geometric Mean: La media geometrica è una metrica che combina la sensibilità e la specificità in un unico valore,
utilizzato per avere un indicatore equilibrato delle prestazioni del modello.

### 7. RISULTATI

Dopo l'esecuzione della valutazione, il programma produce due output principali. Il primo è un file CSV che contiene i
valori delle metriche selezionate, calcolati in base alle predizioni effettuate dal modello. Questo file permette di
analizzare le prestazioni del modello in modo dettagliato e quantitativo. Il secondo output è un plot della matrice di
confusione, salvato come immagine, che fornisce una rappresentazione visiva degli errori e delle corrette classificazioni
effettuate. Questo grafico aiuta a comprendere meglio il comportamento del modello, specialmente in presenza di classi
sbilanciate.


### 8. ESECUZIONE DEL PROGRAMMA

Il punto di ingresso del programma è il file `main.py`. Il sistema utilizza `argparse` per permettere all'utente di personalizzare l'esecuzione tramite riga di comando.

**Parametri disponibili:**
- `--k`: Numero di vicini per il K-NN (default: 3)
- `--method`: Metodo di valutazione da utilizzare. Scelte valide: `holdout`, `kfold`, `loo`
- `--train`: Percentuale del dataset destinata al training
- `--folds`: Numero di partizioni
- `--datase`: Percorso del file CSV da analizzare

**Esempi di esecuzione:**

1. Esecuzione di default (k=3, method=holdout, train=70)

    `docker compose up --> default`

2. Esecuzione con Holdout (80% training, k=5):

    `docker compose run ai-project --k 5 --method holdout --train 80`

3. Esecuzione con K-Fold Cross Validation (10 folds, k=7):

    `docker compose run ai-project --k 7 --method kfold --folds 10`

4. Esecuzione con Leave-One-Out:

    `docker compose run ai-project --method loo`

5.  Data set personalizzato:

    `docker compose run ai-project --k 5 --method holdout --train 70 --dataset dati/version_1.csv`
    

### 9. ISTRUZIONI DOCKER

Il progetto è configurato per essere eseguito all'interno di un container Docker, garantendo un ambiente isolato e riproducibile.
Oltre al comando `docker run` standard, abbiamo incluso un file `docker-compose.yml` per un'esecuzione rapida e automatizzata.

#### Requisiti
- Docker installato sulla macchina.

#### 1. Costruire l'immagine
Aprire il terminale nella cartella radice del progetto ed eseguire:

    docker build -t ai_project .

#### 2. Eseguire il container Docker Compose
Con un solo comando, Docker Compose costruirà l'immagine (se necessario), mapperà in automatico i volumi (lettura da `dati/`, salvataggio in `plots/` e `performances/`) e avvierà il progetto usando i parametri definiti nel file `.env`:

    docker-compose up --build

I parametri di default (come il K o il metodo di valutazione) possono essere modificati semplicemente aprendo e modificando il file `.env` prima di lanciare il comando.

##### 3. Spiegazione dei volumi utilizzati:
- `-v .../dati:/app/dati`: Passa il dataset locale (lettura) al container.
- `-v .../plots:/app/plots`: Permette al container di salvare i grafici generati nella cartella locale 'plots'.
- `-v ..../performances:/app/performances`: Permette al container di salvare il file CSV dei risultati finali nella cartella locale 'performances'.


### 10. Conclusione

In conclusione, questo progetto offre un ambiente potente e interattivo per la classificazione dei dati medici, con un focus particolare sull'uso dell'algoritmo KNN. La struttura del progetto è progettata per garantire un'ampia flessibilità, permettendo agli utenti di personalizzare e ottimizzare il modello a seconda delle esigenze specifiche del loro dataset.  
Ogni fase del processo, dal preprocessing dei dati all'addestramento del modello, fino alla validazione, è pensata per offrire una solida base di lavoro che consenta di ottenere risultati accurati e significativi.  

Nel complesso, il progetto fornisce un workflow completo e strutturato, che aiuta non solo a sviluppare modelli di classificazione efficaci, ma anche a comprenderne a fondo il comportamento e le prestazioni. Questo è particolarmente importante in ambito medico, dove la precisione e l'affidabilità delle previsioni possono avere un impatto diretto sulla diagnosi e sul trattamento dei pazienti.
