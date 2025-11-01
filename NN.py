import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import create_dataloaders
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SimpleNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 128 * 128, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, return_preds=False):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            if return_preds:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    
    if return_preds:
        return epoch_loss, epoch_acc, all_preds, all_labels
    return epoch_loss, epoch_acc

def main():
    DATASET_ROOT = './Data'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 120
    MODEL_SAVE_PATH = './best_nn_model.pth'

    train_loader, val_loader, test_loader, class_names = create_dataloaders(DATASET_ROOT, batch_size=BATCH_SIZE)
    if not train_loader:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    model = SimpleNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    
    print("\n--- Rozpoczęcie treningu (Sieć Neuronowa) ---")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoka {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Czas: {epoch_duration:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Zapisano nowy najlepszy model z dokładnością: {best_val_acc:.4f}")

    print("\n--- Zakończono trening ---")

    print("\n--- Testowanie najlepszego modelu na zbiorze testowym ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device, return_preds=True)
    print(f"Wyniki na zbiorze testowym -> Strata: {test_loss:.4f}, Dokładność: {test_acc:.4f}")

    print("\n--- Raport Klasyfikacji ---")
    report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    print(classification_report(test_labels, test_preds, target_names=class_names))
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T
    plt.figure(figsize=(10, len(class_names) + 4))
    sns.heatmap(report_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Raport Klasyfikacji')
    plt.show()

    print("\n--- Macierz Pomyłek ---")
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Przewidziana etykieta')
    plt.ylabel('Prawdziwa etykieta')
    plt.title('Macierz Pomyłek')
    plt.show()

if __name__ == '__main__':
    main()
