import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import numpy as np
import time

from preprocessing import create_dataloaders
from CNN import SimpleCNN, train_epoch, evaluate
from NN import SimpleNN

def grid_search_pytorch_model(model_class, param_grid, train_loader, test_loader, device, class_names, model_name):
    """
    Przeprowadza Grid Search dla danego modelu PyTorch przy użyciu skorch i GridSearchCV.
    """
    print(f"--- Rozpoczęcie Grid Search dla {model_name} ---")
    start_time = time.time()

    # Użycie NeuralNetClassifier z skorch jako wrappera dla modelu PyTorch
    net = NeuralNetClassifier(
        module=model_class,
        module__num_classes=len(class_names),
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        device=device,
        max_epochs=5,  # Ograniczenie epok dla szybkości Grid Search
        verbose=0,     # Wyłączenie logowania skorch
        train_split=None # Używamy już podzielonego zbioru walidacyjnego z create_dataloaders
    )

    # Konfiguracja GridSearchCV
    gs = GridSearchCV(
        estimator=net, 
        param_grid=param_grid, 
        cv=3,  # 3-krotna walidacja krzyżowa
        scoring='accuracy', 
        verbose=2,
        n_jobs=-1 # Użyj wszystkich dostępnych rdzeni procesora
    )

    # Konwersja danych do formatu akceptowalnego przez scikit-learn (NumPy)
    # Skorch wymaga danych w formacie X, y
    X_train = np.vstack([batch[0].numpy() for batch in train_loader])
    y_train = np.concatenate([batch[1].numpy() for batch in train_loader])

    # Uruchomienie Grid Search
    gs.fit(X_train, y_train)

    end_time = time.time()
    print(f"--- Zakończono Grid Search dla {model_name} w {(end_time - start_time):.2f}s ---")

    print(f"Najlepsze parametry dla {model_name}: {gs.best_params_}")
    print(f"Najlepsza dokładność walidacyjna dla {model_name}: {gs.best_score_:.4f}")

    # Ewaluacja na zbiorze testowym
    X_test = np.vstack([batch[0].numpy() for batch in test_loader])
    y_test = np.concatenate([batch[1].numpy() for batch in test_loader])
    
    test_acc = gs.score(X_test, y_test)
    print(f"Dokładność na zbiorze testowym z najlepszymi parametrami: {test_acc:.4f}\n")

    return gs.best_params_, gs.cv_results_

def main():
    DATASET_ROOT = './Data'
    BATCH_SIZE = 32

    # Używamy większego batcha do przygotowania danych, aby było szybciej
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        DATASET_ROOT, 
        batch_size=128, 
        train_split=0.7, 
        val_split=0.15
    )
    if not train_loader:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # --- Grid Search dla SimpleNN ---
    nn_param_grid = {
        'lr': [0.001, 0.0001],
        'module__hidden_size': [100, 200],
        'optimizer': [optim.Adam, optim.SGD]
    }
    grid_search_pytorch_model(SimpleNN, nn_param_grid, train_loader, test_loader, device, class_names, "SimpleNN")

    # --- Grid Search dla SimpleCNN ---
    cnn_param_grid = {
        'lr': [0.001, 0.0001],
        'module__fc_size': [80, 120],
        'module__dropout_rate': [0.5, 0.3]
    }
    # Dla CNN używamy val_loader jako zbioru testowego, aby przyspieszyć
    grid_search_pytorch_model(SimpleCNN, cnn_param_grid, train_loader, val_loader, device, class_names, "SimpleCNN")


if __name__ == '__main__':
    main()
