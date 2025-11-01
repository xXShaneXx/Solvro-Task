import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os

def get_mean_std(loader: DataLoader):
    mean = 0.
    std = 0.
    total_images_count = 0
    print("Obliczanie średniej i odchylenia standardowego na zbiorze treningowym...")
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    print("Obliczanie zakończone.")
    return mean, std

def create_dataloaders(dataset_path: str, batch_size: int = 32, train_split: float = 0.7, val_split: float = 0.15):
    if not os.path.isdir(dataset_path):
        print(f"BŁĄD: Folder ze zbiorem danych nie został znaleziony pod adresem '{dataset_path}'")
        return None, None, None, None

    initial_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=dataset_path, transform=initial_transform)
    class_names = dataset.classes
    print(f"Znaleziono {len(dataset)} obrazów w {len(class_names)} klasach.")

    test_split = 1 - train_split - val_split
    if test_split < 0:
        raise ValueError("Suma podziałów (train_split, val_split) nie może przekraczać 1.0")

    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Podział danych: Treningowe: {len(train_dataset)}, Walidacyjne: {len(val_dataset)}, Testowe: {len(test_dataset)}")


    temp_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    mean, std = get_mean_std(temp_train_loader)
    print(f"Obliczona średnia: {mean.tolist()}")
    print(f"Obliczone odchylenie standardowe: {std.tolist()}")

    train_transforms = transforms.Compose([
        transforms.RandomCrop(128, padding=4),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])

    # Tworzenie klasy w celu przpisania transformacji do augumentacji danych
    class TransformedDataset(Subset):
        def __init__(self, subset, transform=None):
            super().__init__(subset.dataset, subset.indices)
            self.transform = transform

        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            if self.transform:
                x = self.transform(x)
            return x, y

    train_dataset = TransformedDataset(train_dataset, train_transforms)
    val_dataset = TransformedDataset(val_dataset, val_test_transforms)
    test_dataset = TransformedDataset(test_dataset, val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("\nDataLoadery zostały pomyślnie utworzone.")
    return train_loader, val_loader, test_loader, class_names

if __name__ == '__main__':
    DATASET_ROOT = './Data'
    
    train_loader, val_loader, test_loader, class_names = create_dataloaders(DATASET_ROOT)

    if train_loader:
        print("\nSprawdzanie paczki danych z train_loader...")
        images, labels = next(iter(train_loader))
        print(f"Rozmiar paczki obrazów: {images.shape}") 
        print(f"Rozmiar paczki etykiet: {labels.shape}")   
        print(f"Przykładowa etykieta: {labels[0].item()} ({class_names[labels[0]]})")
        print(f"Min/Max wartości pikseli po normalizacji: {images.min():.2f}/{images.max():.2f}")
