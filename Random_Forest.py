import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from preprocessing import create_dataloaders

def get_data_from_loader(loader):
    features = []
    labels = []
    
    print(f"Przetwarzanie danych z {type(loader.dataset).__name__}...")
    for images, lbls in loader:
        # Spłaszczenie obrazów z (batch, channels, height, width) do (batch, features)
        images = images.view(images.shape[0], -1)
        features.append(images.numpy())
        labels.append(lbls.numpy())
        
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print("Przetwarzanie zakończone.")
    return features, labels

def main():
    DATASET_ROOT = './Data'
    BATCH_SIZE = 64 

    train_loader, _, test_loader, class_names = create_dataloaders(
        DATASET_ROOT, 
        batch_size=BATCH_SIZE,
        train_split=0.8, 
        val_split=0.0
    )

    if not train_loader:
        return

    # Konwersja danych z DataLoaderów do formatu NumPy dla Scikit-learn
    X_train, y_train = get_data_from_loader(train_loader)
    X_test, y_test = get_data_from_loader(test_loader)

    print(f"\nRozmiar danych treningowych: {X_train.shape}")
    print(f"Rozmiar danych testowych: {X_test.shape}")

    print("\n--- Rozpoczęcie trenowania Random Forest ---")
    start_time = time.time()
    

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
    
    rf_classifier.fit(X_train, y_train)
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"--- Zakończono trening w {training_duration:.2f}s ---")
    print(f"Dokładność OOB (Out-of-Bag): {rf_classifier.oob_score_:.4f}")


    print("\n--- Testowanie modelu na zbiorze testowym ---")
    start_time = time.time()
    y_pred = rf_classifier.predict(X_test)
    end_time = time.time()
    
    test_acc = np.mean(y_pred == y_test)
    print(f"Czas predykcji: {(end_time - start_time):.2f}s")
    print(f"Dokładność na zbiorze testowym: {test_acc:.4f}")

    print("\n--- Raport Klasyfikacji ---")
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T
    plt.figure(figsize=(10, len(class_names) + 4))
    sns.heatmap(report_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Raport Klasyfikacji - Random Forest')
    plt.show()

    print("\n--- Macierz Pomyłek ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Przewidziana etykieta')
    plt.ylabel('Prawdziwa etykieta')
    plt.title('Macierz Pomyłek - Random Forest')
    plt.show()

if __name__ == '__main__':
    main()
