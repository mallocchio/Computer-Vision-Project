import os
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa

# Definisci la cartella di input e output
input_folder = 'C:/Users/Stefano Monte/Desktop/Boat dataset/VID-20240904-WA0004'
output_folder = 'C:/Users/Stefano Monte/Desktop/Boat dataset/VID-20240904-WA0004_augmented'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Configurazione delle trasformazioni di PyTorch
transform = transforms.Compose([
    transforms.RandomRotation(40),  # Ruota le immagini fino a 40 gradi
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),  # Traslazione
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Zoom
    transforms.RandomHorizontalFlip(),  # Ribaltamento orizzontale
    transforms.ToTensor()  # Converte l'immagine in un tensor
])

# Configurazione degli effetti aggiuntivi (blur, ecc.) con imgaug
augmenters = iaa.Sequential([
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.5))),  # Applica blur gaussiano al 30% delle immagini
    iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),  # Aggiungi rumore gaussiano
    iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2))),  # Cambia leggermente la luminosità
    iaa.Sometimes(0.3, iaa.Affine(rotate=(-20, 20)))  # Rotazioni aggiuntive
])

# Carica le immagini e applica l'augmentation
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path)  # Carica l'immagine
    img = transform(img)  # Applica le trasformazioni di PyTorch
    img = img.numpy()  # Converti l'immagine in un array NumPy
    img = np.moveaxis(img, 0, -1)  # Riordina gli assi per imgaug (HWC)

    # Applica imgaug
    augmented_images = augmenters(images=np.array([img]))  # Applica gli effetti di imgaug

    # Salva le immagini augmentate
    for i in range(5):  # Genera 5 immagini augmentate per ogni immagine originale
        augmented_image = augmented_images[i]
        augmented_image_pil = Image.fromarray(augmented_image)  # Converti l'array in immagine
        augmented_image_pil.save(os.path.join(output_folder, f'aug_{i}_{img_name}'))  # Salva l'immagine augmentata
