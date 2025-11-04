# Klasyfikacja Obrazów - Zadanie Aplikacyjne

Repozytorium zawiera implementację i porównanie trzech różnych modeli do klasyfikacji obrazów na 10 klasach.

## Struktura Projektu

- `Data/`: Folder zawierający zbiór danych ([Simple hand-drawn and digitized images](https://www.kaggle.com/datasets/gergvincze/simple-hand-drawn-and-digitized-images/data)).
- `CNN.py`: Skrypt do trenowania i oceny modelu konwolucyjnej sieci neuronowej (CNN).
- `NN.py`: Skrypt do trenowania i oceny prostego modelu sieci neuronowej (NN).
- `Random_Forest.py`: Skrypt do trenowania i oceny modelu lasu losowego (Random Forest).
- `Zadanie.ipynb`: Notatnik Jupyter zawierający cały proces od analizy danych po trening i ewaluację modeli.
- `preprocessing.py`: Moduł do przygotowywania i ładowania danych.
- `datasetStatistic.py`: Skrypt do analizy i wizualizacji statystyk zbioru danych.
- `visualize_transforms.py`: Skrypt do wizualizacji transformacji na obrazach.
- `requirements.txt`: Lista zależności Pythona.
- `best_model.pth`: Zapisane wagi przykładowego modelu CNN.
- `best_nn_model.pth`: Zapisane wagi przykładowego modelu NN.

## 1. Wymagania Wstępne

- Python 3.8+
- `pip` i `venv`

## 2. Instalacja

1.  **Sklonuj repozytorium:**
    ```bash
    git clone https://github.com/xXShaneXx/Solvro-Task.git
    cd Solvro-Task
    ```

2.  **Utwórz i aktywuj wirtualne środowisko:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    *W systemie Windows użyj:* `venv\Scripts\activate`

3.  **Zainstaluj wymagane biblioteki:**
    ```bash
    pip install -r requirements.txt
    ```

## 3. Przygotowanie Zbioru Danych

Model wymaga zbioru danych umieszczonego w folderze `Data/`. Każda klasa obrazów powinna znajdować się w osobnym podfolderze wewnątrz `Data/`.

Struktura folderu `Data` powinna wyglądać następująco:
```
Data/
├── anchor/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── balloon/
│   ├── 1.png
│   └── ...
├── bicycle/
└── ... (pozostałe 7 klas)
```

## 4. Użycie

Możesz uruchomić poszczególne skrypty lub skorzystać z notatnika Jupyter.

### A. Uruchamianie Skryptów

Wszystkie skrypty należy uruchamiać z głównego folderu repozytorium.

1.  **Analiza zbioru danych:**
    Aby przeanalizować i zwizualizować statystyki zbioru danych, uruchom:
    ```bash
    python datasetStatistic.py
    ```

2.  **Trening modeli:**
    - **CNN:**
      ```bash
      python CNN.py
      ```
    - **Sieć neuronowa (NN):**
      ```bash
      python NN.py
      ```
    - **Random Forest:**
      ```bash
      python Random_Forest.py
      ```
    Po zakończeniu treningu, modele zapisują swoje najlepsze wagi (`best_model.pth` lub `best_nn_model.pth`) i wyświetlają raporty klasyfikacji oraz macierze pomyłek.

### B. Notatnik Jupyter (`Zadanie.ipynb`)

Notatnik `Zadanie.ipynb` zawiera kompletny przepływ pracy: od analizy danych, przez preprocessing, po trening i ewaluację wszystkich trzech modeli.

1.  **Uruchom serwer Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  W przeglądarce otwórz plik `Zadanie.ipynb`.
3.  Wykonuj komórki po kolei, aby zobaczyć cały proces.
