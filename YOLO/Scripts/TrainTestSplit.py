import os
import random
import shutil
from glob import glob

# Imposta i percorsi delle cartelle
dataset_dir = 'C:/Users/Stefano Monte/Desktop/Dataset3'  # Cambia con il percorso del tuo dataset
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Creazione delle cartelle per il training e test set
train_images_dir = os.path.join(dataset_dir, 'train', 'images')
train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
test_images_dir = os.path.join(dataset_dir, 'test', 'images')
test_labels_dir = os.path.join(dataset_dir, 'test', 'labels')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Prendi tutte le immagini in formato .jpg e .jpeg e mischiale con un seme per avere casualità ripetibile
image_files = glob(os.path.join(images_dir, '*.jpg')) + glob(os.path.join(images_dir, '*.jpeg'))

# Imposta un seme per garantire che la randomizzazione sia realmente casuale e ripetibile
random.seed(42)
random.shuffle(image_files)

# Imposta il rapporto di divisione
train_ratio = 0.8
split_index = int(len(image_files) * train_ratio)

# Dividi il dataset
train_files = image_files[:split_index]
test_files = image_files[split_index:]

def move_files(image_list, dest_images_dir, dest_labels_dir):
    for image_file in image_list:
        # Nome del file immagine
        image_name = os.path.basename(image_file)

        # Gestisci sia l'estensione .jpg che .jpeg
        if image_name.endswith('.jpg'):
            label_file = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))
        elif image_name.endswith('.jpeg'):
            label_file = os.path.join(labels_dir, image_name.replace('.jpeg', '.txt'))
        
        # Sposta l'immagine
        shutil.copy(image_file, os.path.join(dest_images_dir, image_name))
        
        # Sposta il label corrispondente
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(dest_labels_dir, os.path.basename(label_file)))

# Muovi i file nelle rispettive cartelle
move_files(train_files, train_images_dir, train_labels_dir)
move_files(test_files, test_images_dir, test_labels_dir)

print("Dataset diviso con successo in training e test set!")
