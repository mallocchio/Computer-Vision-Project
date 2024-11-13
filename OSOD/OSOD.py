import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
import cv2
import os
import tempfile
import subprocess
from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
from PIL import Image


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, tasks=["bbox"], distributed=False, output_dir=output_folder)

class DetectronModel:

    def __init__(self, config_file, num_classes, confidence_threshold, learning_rate, max_iter, model_dir="output"):
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # Soglia di confidenza per il rilevamento
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Numero di classi nel dataset

        # Impostazioni di Solver per Few-Shot Learning
        self.cfg.SOLVER.IMS_PER_BATCH = 4  # Batch size
        self.cfg.SOLVER.BASE_LR = learning_rate  # Learning rate
        self.cfg.SOLVER.MAX_ITER = max_iter  # Numero di iterazioni
        self.cfg.SOLVER.STEPS = []  # Nessun ridimensionamento del learning rate
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Batch size per immagine

        # Cartella per salvare il modello
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def register_datasets(self, train_name, val_name, train_annotations, train_images, val_annotations, val_images):
        
        register_coco_instances(train_name, {}, train_annotations, train_images)
        register_coco_instances(val_name, {}, val_annotations, val_images)

        self.cfg.DATASETS.TRAIN = (train_name,)
        self.cfg.DATASETS.TEST = (val_name,)

    def train(self):
        """
        Avvia l'addestramento del modello con il trainer di Detectron2.
        """
        trainer = CustomTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Salva il modello addestrato
        trainer.checkpointer.save("model_final")

        # Esegui la validazione
        val_results = trainer.test(self.cfg, trainer.model)

        print("Validation Results:", val_results)

    def load_trained_model(self, model_weights_path):
        """
        Carica i pesi di un modello pre-addestrato.
        """
        if os.path.exists(model_weights_path):
            self.cfg.MODEL.WEIGHTS = model_weights_path
        else:
            raise FileNotFoundError(f"Il file {model_weights_path} non esiste.")

    def create_predictor(self):
        """
        Crea il predictor con il modello addestrato.
        """
        return DefaultPredictor(self.cfg)
    
    def visualize_predictions(self, image_path, predictor, output_path=None):
        
        """
        Visualizza le predizioni su un'immagine e salva il risultato con un nuovo nome.
        """
        im = cv2.imread(image_path)
        outputs = predictor(im)

        print(outputs)

        # Visualizza le predizioni
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Estrai la cartella e il nome file originale
        folder, file_name = os.path.split(image_path)
        name, ext = os.path.splitext(file_name)

        # Crea il nuovo nome aggiungendo "(bbxs)" al nome originale
        new_file_name = f"{name}(bbxs){ext}"

        # Se output_path è None, salva nella stessa cartella dell'immagine originale
        if output_path is None:
            output_path = os.path.join(folder, new_file_name)

        # Salva l'immagine con il nuovo nome
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

        print(f"Saved prediction image as: {output_path}")

    def process_video(self, input_video_path, output_video_path, predictor, confidence_threshold=0.00):
        """Elabora un video usando MoviePy, applica il modello e salva il video con le predizioni mantenendo risoluzione e frame rate."""
        
        # Funzione per elaborare ogni frame del video
        def process_frame(frame):
            # Converti il frame in un formato compatibile con il modello
            frame_rgb = np.array(frame)  # MoviePy fornisce frame in RGB (già compatibile)
            
            # Applica il modello al frame corrente
            outputs = predictor(frame_rgb)
            
            # Filtra le predizioni in base alla confidence_threshold
            instances = outputs["instances"].to("cpu")
            high_confidence_indices = instances.scores > confidence_threshold
            high_confidence_instances = instances[high_confidence_indices]
            
            # Visualizza solo le predizioni con confidenza superiore alla soglia
            v = Visualizer(frame_rgb, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out_frame = v.draw_instance_predictions(high_confidence_instances).get_image()
            
            # Mantieni la risoluzione originale
            out_frame = Image.fromarray(out_frame).resize((frame.shape[0], frame.shape[1]), Image.Resampling.LANCZOS)
            
            return np.array(out_frame)  # Restituisci il frame processato (con predizioni e bounding box)

        # Apri il video con MoviePy
        video_clip = VideoFileClip(input_video_path)
        
        # Applica la funzione `process_frame` a ciascun frame del video
        processed_clip = video_clip.fl_image(process_frame)
        
        # Mantieni il frame rate originale
        original_fps = video_clip.fps
        
        # Scrivi il video di output mantenendo lo stesso frame rate e risoluzione
        processed_clip.write_videofile(output_video_path, codec='libx264', fps=original_fps)

        # Rilascia le risorse di MoviePy (facoltativo, MoviePy gestisce automaticamente)
        video_clip.reader.close()
        video_clip.audio.reader.close_proc()

    def build_evaluator(self):
        return COCOEvaluator(self.cfg.DATASETS.TEST[0], output_dir=self.model_dir)


def main(train_model=True, video_path=None):
    # Percorsi del dataset e delle annotazioni
    train_name = "boat_train"
    val_name = "boat_val"
    train_annotations = "boat_dataset/annotations/annotations_train.json"
    train_images = "boat_dataset/train"
    val_annotations = "boat_dataset/annotations/annotations_val.json"
    val_images = "boat_dataset/val"
    
    # Configurazione del modello
    config_file = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    num_classes = 1  # Modifica in base al tuo dataset
    confidence_threshold = 0.5
    learning_rate = 0.00025
    max_iter = 300

    # Inizializzazione del modello
    model = DetectronModel(config_file, num_classes, confidence_threshold, learning_rate, max_iter)

    model.register_datasets(train_name, val_name, train_annotations, train_images, val_annotations, val_images)
    
    if train_model:

        model.train()
    else:
      
        model_weights_path = os.path.join(model.model_dir, "model_final.pth")
        model.load_trained_model(model_weights_path)
        

        predictor = model.create_predictor()
        
        if video_path:

            output_video_path = "test_media/WhatsApp Video 2024-09-26 at 11.53.00(300iter).mp4"
            model.process_video(video_path, output_video_path, predictor, 0.85)
        else:
          
            test_image_path = "test_images/VID-20240904-WA0010_out0005.jpg"  # Modifica questo percorso con l'immagine di test
            model.visualize_predictions(test_image_path, predictor)

if __name__ == "__main__":
    # Cambia train_model a False se vuoi solo valutare il modello
    video_file = "test_media/WhatsApp Video 2024-09-26 at 11.53.00.mp4"  # Specifica qui il percorso del video
    main(train_model=False, video_path=video_file)
