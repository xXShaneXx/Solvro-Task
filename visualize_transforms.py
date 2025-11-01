import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

def visualize_transformations(image_path: str):
    if not os.path.exists(image_path):
        print(f"BŁĄD: Plik obrazu nie został znaleziony pod adresem '{image_path}'")
        return

    original_img = Image.open(image_path).convert("RGB")


    transformations = [
        ("Oryginał", None),
        ("Resize(128)", transforms.Resize(128)),
        ("CenterCrop(128)", transforms.CenterCrop(128)),
        ("Grayscale", transforms.Grayscale(num_output_channels=1)),
        ("RandomResizedCrop",transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0))),
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomRotation(15)", transforms.RandomRotation(15)),
        ("RandomCrop", transforms.RandomCrop(128, padding=4))
    ]

    num_transformations = len(transformations)
    fig, axes = plt.subplots(1, num_transformations, figsize=(20, 5))
    fig.suptitle(f'Wizualizacja transformacji dla: {os.path.basename(image_path)}', fontsize=16)

    img_to_transform = original_img
    for i, (name, trans) in enumerate(transformations):
        ax = axes[i]
        
        if trans is not None:

            current_mode = img_to_transform.mode
            img_to_transform = trans(img_to_transform)
            if current_mode == 'RGB' and img_to_transform.mode == 'L':
                pass
        
        ax.set_title(name)
        ax.axis('off')

        # Jeśli obraz jest w skali szarości (tryb 'L'), użyj mapy kolorów 'gray'
        if img_to_transform.mode == 'L':
            ax.imshow(img_to_transform, cmap='gray')
        else:
            ax.imshow(img_to_transform)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def find_random_image(root_dir: str) -> str:
    all_images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(dirpath, filename))
    
    if not all_images:
        return None
        
    return random.choice(all_images)

if __name__ == '__main__':
    DATASET_ROOT = './Data'
    
    random_image_path = find_random_image(DATASET_ROOT)
    
    if random_image_path:
        print(f"Wybrano losowy obraz do wizualizacji: {random_image_path}")
        visualize_transformations(random_image_path)
    else:
        print(f"Nie znaleziono żadnych obrazów w folderze {DATASET_ROOT}")
