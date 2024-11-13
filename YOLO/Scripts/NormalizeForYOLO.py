import json
import os
from PIL import Image  # Libreria Pillow per lavorare con le immagini

# Percorso del file JSON
json_file_path = 'C:/Users/Stefano Monte/Desktop/Boat dataset/VID-20240904-WA0004_augmented/VID-20240904-WA0004_augmented_annotations_json.json'

# Cartella in cui si trovano le immagini
image_folder = 'C:/Users/Stefano Monte/Desktop/Boat dataset/VID-20240904-WA0004_augmented'

# Cartella in cui salvare i file YOLO .txt
output_folder = 'C:/Users/Stefano Monte/Desktop/Boat dataset/VID-20240904-WA0004_augmented/labels'

# Crea la cartella di output se non esiste
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Funzione per convertire il bounding box in formato YOLO
def convert_bbox_to_yolo(size, bbox):
    """Converte le coordinate del bounding box nel formato YOLO"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = bbox[0] + bbox[2] / 2.0  # x_center = x + width / 2
    y = bbox[1] + bbox[3] / 2.0  # y_center = y + height / 2
    w = bbox[2]  # width
    h = bbox[3]  # height
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)

# Carica il file JSON
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Loop attraverso tutte le immagini nel file JSON
for key, value in data.items():
    filename = value['filename']
    regions = value['regions']
    
    # Ottieni il percorso completo dell'immagine
    image_path = os.path.join(image_folder, filename)
    
    # Apri l'immagine e ottieni le dimensioni
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    
    # Crea il file di output corrispondente per YOLO
    yolo_filename = os.path.splitext(filename)[0] + '.txt'
    yolo_file_path = os.path.join(output_folder, yolo_filename)
    
    with open(yolo_file_path, 'w') as yolo_file:
        for region in regions:
            shape_attributes = region['shape_attributes']
            x = shape_attributes['x']
            y = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']
            
            # Converti il bounding box in formato YOLO utilizzando le dimensioni dell'immagine
            bbox = convert_bbox_to_yolo((image_width, image_height), [x, y, width, height])
            
            # Classe dell'oggetto (0 = "boat", perché è l'unica classe presente)
            class_id = 0
            
            # Scrivi nel file di annotazione YOLO nel formato richiesto
            yolo_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

print("Conversione completata!")
