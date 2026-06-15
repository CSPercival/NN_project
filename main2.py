import os
import glob
import json
import re
import argparse
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split

from config.consts import device, lr, weight_decay, epochs, batch_size, img_size, S, B, C, available_classes, conf_threshold
from config.paths import IMG_DIR_VAL, ANN_FILE_VAL
from models.yolo_v1.yolo import YOLOv1
from models.yolo_v1.yolo_architecture import YOLO_architecture
from models.yolo_v1.yolo_loss import YOLO_loss
from training.train import model_train, model_evaluate
from coco.dataloader.transform import yolo_transform 
from coco.dataloader.dataset import COCODataset 

# -------------------------------------------------------------------------
# PARSOWANIE ARGUMENTÓW Z KONSOLI
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Skrypt treningowy YOLOv1")
parser.add_argument("--weights", type=str, default=None, help="Ścieżka do wag początkowych (opcjonalnie)")
parser.add_argument("--results_dir", type=str, default=None, help="Ścieżka do folderu na wyniki (opcjonalnie)")
args = parser.parse_args()

# 1. Ustalanie folderu na wyniki
if args.results_dir:
    RESULTS_DIR = args.results_dir
else:
    i = 1
    while os.path.exists(f"results{i}"):
        i += 1
    RESULTS_DIR = f"results{i}"

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"--> [INFO] Wyniki będą zapisywane w folderze: {RESULTS_DIR}")

# -------------------------------------------------------------------------
# INICJALIZACJA DANYCH
# -------------------------------------------------------------------------
transform = yolo_transform

coco_dataset = COCODataset(IMG_DIR_VAL, ANN_FILE_VAL, transform, img_size, S, B, C, available_classes, True)

train_size = int(0.9 * len(coco_dataset))
val_size = len(coco_dataset) - train_size

train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = YOLOv1(YOLO_architecture).to(device)

# -------------------------------------------------------------------------
# WCZYTYWANIE WAG (Opcjonalne)
# -------------------------------------------------------------------------
if args.weights and os.path.exists(args.weights):
    print(f"--> [LOAD] Wczytuję wagi początkowe z pliku: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
elif args.weights and not os.path.exists(args.weights):
    print(f"--> [WARNING] Podany plik wag '{args.weights}' nie istnieje. Rozpoczynam trening od zera.")
else:
    print("--> [LOAD] Rozpoczynam trening modelu od zera (brak podanych wag).")

# -------------------------------------------------------------------------
# TRENING
# -------------------------------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

train_losses = []
val_losses = []

torch.cuda.empty_cache()
for epoch in range(0, 2000): 
    print(f"Started epoch: {epoch}", flush=True)
    
    train_loss = model_train(model, train_dataloader, optimizer, device, YOLO_loss)
    train_losses.append(train_loss)
    
    val_loss = model_evaluate(model, val_dataloader, device, YOLO_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{RESULTS_DIR}/yolo_weights_epoch_{epoch}.pth")
    
    # Zapis plików JSON (nadpisywanie, żeby nie tworzyć tysięcy plików na dysku)
    with open(f"{RESULTS_DIR}/train_losses_current_epoch_{epoch}.json", "w") as f:
        json.dump([float(loss) for loss in train_losses], f)
        
    with open(f"{RESULTS_DIR}/val_losses_current_epoch_{epoch}.json", "w") as f:
        json.dump([float(loss) for loss in val_losses], f)
        
    print(f"\n  Epoch {epoch} done\n", flush=True)

# -------------------------------------------------------------------------
# GENEROWANIE NAZWY DLA OSTATECZNYCH WAG I STRAT
# -------------------------------------------------------------------------
weight_files = glob.glob(f"{RESULTS_DIR}/yolo_weights_final_*.pth")

max_idx = 0
for f in weight_files:
    match = re.search(r'yolo_weights_final_(\d+)\.pth', f)
    if match:
        max_idx = max(max_idx, int(match.group(1)))

next_idx = max_idx + 1

new_weights_path = f"{RESULTS_DIR}/yolo_weights_final_{next_idx}.pth"
torch.save(model.state_dict(), new_weights_path)
print(f"--> [AUTO-SAVE] Trening zakończony. Zapisano ostateczne wagi do: {new_weights_path}")

new_train_losses_path = f"{RESULTS_DIR}/train_losses_final_{next_idx}.json"
with open(new_train_losses_path, "w") as f:
    json.dump([float(loss) for loss in train_losses], f)
    
new_val_losses_path = f"{RESULTS_DIR}/val_losses_final_{next_idx}.json"
with open(new_val_losses_path, "w") as f:
    json.dump([float(loss) for loss in val_losses], f)
    
print(f"--> [AUTO-SAVE] Ostateczne wartości błędów zapisane do: {new_train_losses_path} i {new_val_losses_path}")