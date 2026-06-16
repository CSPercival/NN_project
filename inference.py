import torch
import argparse
import glob
import os
from torch.utils.data import DataLoader

from config.consts import img_size, S, B, C, available_classes
from config.paths import IMG_DIR_VAL, ANN_FILE_VAL
from models.yolo_v1.yolo import YOLOv1
from models.yolo_v1.yolo_architecture import YOLO_architecture
from coco.dataloader.transform import yolo_transform 
from coco.dataloader.dataset import COCODataset 
from utils.visualize import visualize_yolo
from utils.best_box import select_best_box

def run_inference(weights_path, threshold, num_images):
    # Ladowanie CAŁEGO zbioru walidacyjnego (bez random_split)
    transform = yolo_transform
    val_dataset = COCODataset(IMG_DIR_VAL, ANN_FILE_VAL, transform, img_size, S, B, C, available_classes, True)

    # Batch size 1 ułatwia pojedynczą wizualizację
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Inicjalizacja modelu i wczytywanie wag
    model = YOLOv1(YOLO_architecture)

    # Automatyczne szukanie najnowszych wag (uwzględnia foldery results*/ z Twojego main.py)
    if weights_path == "auto":
        weight_files = glob.glob("yolo_weights*.pth") + glob.glob("results*/yolo_weights*.pth")
        if not weight_files:
            print("Błąd: Nie znaleziono żadnego pliku z wagami (yolo_weights*.pth) w głównym folderze ani w folderach results.")
            return
        weights_path = max(weight_files, key=os.path.getmtime)
        print(f"--> [AUTO] Najnowsze wagi znalezione to: {weights_path}")

    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Sukces: Pomyślnie załadowano wagi z {weights_path}")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {weights_path}. Najpierw uruchom main.py!")
        return
        
    model.eval()

    print(f"Docelowy próg pewności (conf_threshold) = {threshold}")

    images_shown = 0
    with torch.no_grad():
        for images, targets in val_dataloader:
            if images_shown >= num_images:
                break
            
            # Predykcja modelu
            
            predictions = model(images)
            # Wybranie najlepszych ramek z przewidywań
            best_boxes = select_best_box(predictions[0])
            
            # --- DYNAMICZNY THRESHOLD ---
            # Znajdujemy największą pewność (confidence) w całym wyjściu modelu dla tego obrazka.
            # W YOLOv1 confidence jest zazwyczaj pod indeksem 0 w wymiarze cech.
            max_conf = torch.max(best_boxes[..., 4]).item()
            

            
            print("Predictions (Przewidywane):")
            visualize_yolo(images[0], best_boxes, S, C, val_dataset.id_to_category_name, conf_threshold=threshold)


            
            
            print(f"\n--- Obraz {images_shown + 1} ---")
            
            # Wizualizacja prawdziwych ramek (Ground Truth)
            print("Ground Truth (Oczekiwane):")
            visualize_yolo(images[0], targets[0], S, C, val_dataset.id_to_category_name, conf_threshold=threshold)
            
            images_shown += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zależności inferencji i wizualizacji (YOLOv1)")
    
    # Argumenty, które możesz zmieniać w terminalu
    parser.add_argument("--weights", type=str, default="auto", help="Ścieżka do zapisanych wag modelu (domyślnie 'auto')")
    parser.add_argument("--threshold", type=float, default=0.2, help="Docelowy próg pewności predykcji")
    parser.add_argument("--num_images", type=int, default=2, help="Liczba obrazów do wyświetlenia")
    
    args = parser.parse_args()

    run_inference(args.weights, args.threshold, args.num_images)