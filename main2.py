import os
import glob
import json
import re
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split

from config.consts import device, lr, weight_decay, epochs, batch_size, img_size, S, B, C, available_classes, conf_threshold
from config.paths import IMG_DIR, ANN_FILE
from models.yolo_v1.yolo import YOLOv1
from models.yolo_v1.yolo_architecture import YOLO_architecture
from models.yolo_v1.yolo_loss import YOLO_loss
from training.train import model_train
from coco.dataloader.transform import yolo_transform 
from coco.dataloader.dataset import COCODataset 

transform = yolo_transform

coco_dataset = COCODataset(IMG_DIR, ANN_FILE, transform, img_size, S, B, C, available_classes, True)

train_size = int(0.9 * len(coco_dataset))
val_size = len(coco_dataset) - train_size

train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = YOLOv1(YOLO_architecture).to(device)

# -------------------------------------------------------------------------
# ODCZYTYWANIE NAJNOWSZYCH WAG (Bez ręcznej specyfikacji ścieżki)
# -------------------------------------------------------------------------
weight_files = glob.glob("yolo_weights*.pth")

if not weight_files:
    print("Błąd: Nie znaleziono żadnego pliku z wagami (yolo_weights*.pth) w głównym katalogu.")
    print("Uruchom najpierw main.py, aby stworzyć pierwsze wagi!")
    exit(1)

# Znalezienie pliku, który był modyfikowany najpóźniej
latest_weights = max(weight_files, key=os.path.getmtime)
print(f"--> [AUTO-LOAD] Automatycznie wczytuję najnowsze wagi z pliku: {latest_weights}")

# Wczytywanie state_dict
model.load_state_dict(torch.load(latest_weights, map_location=device))

# -------------------------------------------------------------------------
# TRENING
# -------------------------------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

train_losses = []

torch.cuda.empty_cache()
for epoch in range(500):
    print(f"Started epoch: {epoch} (kontynuacja treningu)", flush=True)
    train_loss = model_train(model, train_dataloader, optimizer, device, YOLO_loss)
    train_losses.append(train_loss)
    torch.save(model.state_dict(), f"yolo_weights_last_training_epoch_{10 + epoch}.pth")
    new_losses_path = f"train_last_training_losses_{epoch}.json"
    with open(new_losses_path, "w") as f:
        json.dump([float(loss) for loss in train_losses], f)
    print(f"\n  Training done\n", flush=True)

# -------------------------------------------------------------------------
# GENEROWANIE NAZWY DLA "KOLEJNYCH" WAG I STRAT
# -------------------------------------------------------------------------
# Szukamy najwyższego numerka dotychczasowych wag (np. yolo_weights_2.pth -> 2)
max_idx = 0
for f in weight_files:
    match = re.search(r'yolo_weights_?(\d*)\.pth', f)
    if match:
        num = match.group(1)
        if num: # jeśli znalazło numerek na końcu
            max_idx = max(max_idx, int(num))
        else: # jeśli to po prostu "yolo_weights.pth"
            max_idx = max(max_idx, 1)

next_idx = max_idx + 1

new_weights_path = f"yolo_weights_{next_idx}.pth"
torch.save(model.state_dict(), new_weights_path)
print(f"--> [AUTO-SAVE] Kontynuacja treningu zakończona. Wypluto KOLEJNE wagi do: {new_weights_path}")

new_losses_path = f"train_losses_{next_idx}.json"
with open(new_losses_path, "w") as f:
    json.dump([float(loss) for loss in train_losses], f)
print(f"--> [AUTO-SAVE] Kolejne wartości błędu opublikowane z treningu zapisane do: {new_losses_path}")
