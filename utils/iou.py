import torch
from config.consts import S, img_size

def iou(boxes1, boxes2):
    # boxes1, boxes2 shapes: [..., 4] -> [x_rel, y_rel, w, h]
    # x_rel, y_rel are relative to the grid cell bounds (0 to 1)
    # w, h are relative to the whole image bounds (0 to 1)
    # Convert x_rel, y_rel to scale relative to the image size
    box1_x_img = boxes1[..., 0:1] / S
    box1_y_img = boxes1[..., 1:2] / S
    box2_x_img = boxes2[..., 0:1] / S
    box2_y_img = boxes2[..., 1:2] / S

    box1_x1 = box1_x_img - boxes1[..., 2:3] / 2
    box1_y1 = box1_y_img - boxes1[..., 3:4] / 2
    box1_x2 = box1_x_img + boxes1[..., 2:3] / 2
    box1_y2 = box1_y_img + boxes1[..., 3:4] / 2

    box2_x1 = box2_x_img - boxes2[..., 2:3] / 2
    box2_y1 = box2_y_img - boxes2[..., 3:4] / 2
    box2_x2 = box2_x_img + boxes2[..., 2:3] / 2
    box2_y2 = box2_y_img + boxes2[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = boxes1[..., 2:3] * boxes1[..., 3:4]
    box2_area = boxes2[..., 2:3] * boxes2[..., 3:4]

    union = box1_area + box2_area - intersection

    return intersection / (union).clamp(1e-6)