import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split

from config.consts import device, lr, weight_decay, epochs, batch_size, img_size, S, B, C, available_classes
from config.paths import IMG_DIR, ANN_FILE
from models.yolo_v1.yolo import YOLOv1
from models.yolo_v1.yolo_architecture import YOLO_architecture
from models.yolo_v1.yolo_loss import YOLO_loss
from training.train import model_train
from coco.dataloader.transform import yolo_transform 
from coco.dataloader.dataset import COCODataset 
from utils.visualize import visualize_yolo
from utils.best_box import select_best_box

transform = yolo_transform

coco_dataset = COCODataset(IMG_DIR, ANN_FILE, transform, img_size, S, B, C, available_classes, True)

train_size = int(0.1 * len(coco_dataset))
val_size = len(coco_dataset) - train_size

train_dataset, val_dataset = random_split(coco_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# for images, targets in val_dataloader:
#     visualize_yolo(images[0].to("cpu"), targets[0].to("cpu"), S, C, coco_dataset.id_to_category_name)

model = YOLOv1(YOLO_architecture).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

train_losses = []
val_losses = []

torch.cuda.empty_cache()
for epoch in range(epochs):
    print(f"Started epoch: {epoch}", flush=True)
    train_loss = model_train(model, train_dataloader, optimizer, device, YOLO_loss)
    train_losses.append(train_loss)
    print(f"\n  Training done\n", flush=True)


i = 0
model = model.to("cpu")
# predictions = model(images.to("cpu"))
for images, targets in train_dataloader:
    visualize_yolo(images[i].to("cpu"), targets[i].to("cpu"), S, C, coco_dataset.id_to_category_name)

    images = images.to(device)
    targets = targets.to(device)
    with torch.no_grad():
      predictions = model(images.to("cpu"))
      print(predictions[i].to("cpu").shape)
      visualize_yolo(images[i].to("cpu"), select_best_box(predictions[i].to("cpu")), S, C, coco_dataset.id_to_category_name)
      print(select_best_box(predictions[i].to("cpu")))
      # break
      i += 1
      if i == 2:
        break