import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import os
from PIL import Image

def target_transform(image, target, S, C, cat_id_to_index):
    w, h = image.size
    grid_w = w / S
    grid_h = h / S

    boxes = target['boxes']
    labels = target['labels']

    new_target = torch.zeros((S, S, C + 5))

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        x_grid = min(S - 1, int(x_center * S / w))
        y_grid = min(S - 1, int(y_center * S / h))

        x_rel = x_center / grid_w - x_grid
        y_rel = y_center / grid_h - y_grid

        new_w = abs(x2 - x1) / w
        new_h = abs(y2 - y1) / h

        new_area = new_w * new_h
        existing_area = new_target[y_grid, x_grid, C+3] * new_target[y_grid, x_grid, C+4]

        label = cat_id_to_index[label.item()]

        if new_target[y_grid, x_grid, C] == 0 or new_area > existing_area:
            new_target[y_grid, x_grid] = 0
            new_target[y_grid, x_grid, label] = 1
            new_target[y_grid, x_grid, C:C+5] = torch.tensor([1, x_rel, y_rel, new_w, new_h])

    return new_target



def yolo_transform(image, target, cat_id_to_index, new_size=(448, 448), S=7, C=80):
    target = target_transform(image, target, S, C, cat_id_to_index)
    w, h = image.size

    image = F.resize(image, new_size)
    image = F.to_tensor(image)

    return image, target