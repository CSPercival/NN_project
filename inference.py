import torch
import argparse
from torch.utils.data import DataLoader, random_split

from config.consts import img_size, S, B, C, available_classes
from config.paths import IMG_DIR, ANN_FILE
from models.yolo_v1.yolo import YOLOv1
from models.yolo_v1.yolo_architecture import YOLO_architecture
from coco.dataloader.transform import yolo_transform 
from coco.dataloader.dataset import COCODataset 
from utils.visualize import visualize_yolo
from utils.best_box import select_best_box

def run_inference(weights_path, threshold, num_images):
    # Ladowanie zbioru danych
    transform = yolo_transform
    coco_dataset = COCODataset(IMG_DIR, ANN_FILE, transform, img_size, S, B, C, available_classes, True)

    # Zapewnienie powtarzalnego podziału (takiego samego jak w main.py)
    # Możesz dodać generator=torch.Generator().manual_seed(23) do random_split, jeśli używasz SEED
    train_size = int(0.1 * len(coco_dataset))
    val_size = len(coco_dataset) - train_size
    _, val_dataset = random_split(coco_dataset, [train_size, val_size])
    
    # Batch size 1 ułatwia pojedynczą wizualizację
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Inicjalizacja modelu i wczytywanie wag
    model = YOLOv1(YOLO_architecture)
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Sukces: Pomyślnie załadowano wagi z {weights_path}")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {weights_path}. Najpierw uruchom main.py!")
        return
        
    model.eval()

    print(f"Używany próg pewności (conf_threshold) = {threshold}")

    images_shown = 0
    with torch.no_grad():
        for images, targets in val_dataloader:
            if images_shown >= num_images:
                break
            
            # Predykcja modelu
            predictions = model(images)
            
            print(f"\n--- Obraz {images_shown + 1} ---")
            
            # Wizualizacja prawdziwych ramek (Ground Truth)
            print("Ground Truth (Oczekiwane):")
            visualize_yolo(images[0], targets[0], S, C, coco_dataset.id_to_category_name, conf_threshold=0.1)
            
            # Wizualizacja predykcji modelu (przepuszczona przez select_best_box i threshold)
            print("Predictions (Przewidywane):")
            best_boxes = select_best_box(predictions[0])
            visualize_yolo(images[0], best_boxes, S, C, coco_dataset.id_to_category_name, conf_threshold=threshold)
            
            images_shown += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zależności inferencji i wizualizacji (YOLOv1)")
    
    # Argumenty, które możesz zmieniać w terminalu
    parser.add_argument("--weights", type=str, default="yolo_weights.pth", help="Ścieżka do zapisanych wag modelu")
    parser.add_argument("--threshold", type=float, default=0.2, help="Próg pewności (confidence threshold) dla predykcji")
    parser.add_argument("--num_images", type=int, default=2, help="Liczba obrazów do wyświetlenia")
    
    args = parser.parse_args()

    run_inference(args.weights, args.threshold, args.num_images)
