import torch
from config.consts import S, img_size

def iou(boxes1, boxes2):
    box1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
    box1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
    box1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
    box1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2

    box2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
    box2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
    box2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
    box2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = boxes1[..., 2:3] * boxes1[..., 3:4]
    box2_area = boxes2[..., 2:3] * boxes2[..., 3:4]

    union = box1_area + box2_area - intersection

    return intersection / (union).clamp(1e-6)