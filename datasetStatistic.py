import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def analyze_dataset(dataset_path: str):
    print(f"--- Analiza zbioru danych w: {dataset_path} ---")

    if not os.path.isdir(dataset_path):
        print(f"\nBŁĄD: Folder ze zbiorem danych nie został znaleziony pod adresem '{dataset_path}'")
        print("Proszę zaktualizować zmienną 'dataset_root' w skrypcie.")
        return

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    try:
        dataset = ImageFolder(root=dataset_path, transform=transform)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nBŁĄD: Nie można załadować zbioru danych. Sprawdź strukturę folderów.")
        print(f"Szczegóły błędu: {e}")
        return
        
    print(f"\nZnaleziono {len(dataset)} obrazów.")

    num_classes = len(dataset.classes)
    class_names = dataset.classes
    print(f"Liczba klas: {num_classes}")
    print(f"Nazwy klas: {class_names}")

    print("\nAnaliza rozkładu klas:")
    class_counts = Counter(dataset.targets)
    most_common = class_counts.most_common()
    print(f"Najliczniejsza klasa: '{class_names[most_common[0][0]]}' ({most_common[0][1]} obrazów)")
    print(f"Najmniej liczna klasa: '{class_names[most_common[-1][0]]}' ({most_common[-1][1]} obrazów)")

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, [class_counts[i] for i in range(num_classes)])
    plt.title('Rozkład klas w zbiorze danych')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba obrazów')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\nWyświetlanie przykładowych obrazów...")
    sample_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(sample_loader))

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Przykładowe obrazy ze zbioru danych', fontsize=16)
    for i, ax in enumerate(axes.flat):
        # Konwersja tensora do obrazu numpy
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Klasa: {class_names[labels[i]]}")
        ax.axis('off')
    plt.show()

    print("\nAnaliza właściwości obrazów (średnia, odch. standardowe)...")
    loader_stats = DataLoader(dataset, batch_size=64)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader_stats:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print(f"Średnia wartość pikseli (RGB): {mean.tolist()}")
    print(f"Odchylenie standardowe pikseli (RGB): {std.tolist()}")
    print("Wizualna inspekcja przykładowych obrazów jest kluczowa do oceny jakości (ostrość, artefakty, oświetlenie).")
    print("\n--- Koniec analizy ---")


if __name__ == '__main__':
    dataset_root = './1'
    
    analyze_dataset(dataset_root)