import torch

from utils.iou import iou
from config.consts import C, S, B, lambda_coord, lambda_noobj

def YOLO_loss(preds, targets):

    preds = preds.reshape(-1, S, S, C + B * 5)
    batch_size = preds.size(0)

    pred_classes = torch.sigmoid(preds[..., :C])
    pred_boxes = preds[..., C:].reshape(-1, S, S, B, 5)
    pred_conf = torch.sigmoid(pred_boxes[..., 0:1])
    pred_x_y = torch.sigmoid(pred_boxes[..., 1:3])
    pred_w_h = torch.sign(pred_boxes[..., 3:5]) * torch.sqrt(torch.abs(pred_boxes[..., 3:5]).clamp(1e-6))

    target_box = targets[..., C+1:C+5].unsqueeze(3)
    target_classes = targets[..., :C]
    exists_box = targets[..., C:C+1].unsqueeze(3)
    target_x_y = target_box[..., 0:2]
    target_w_h = torch.sqrt(target_box[..., 2:4].clamp(1e-6))

    ious = iou(pred_boxes[..., 1:5], target_box)

    max_iou, best_box_id = torch.max(ious, dim=3, keepdim=True)
    best_box_mask = torch.zeros_like(ious).scatter_(3, best_box_id, 1)

    loss_x_y = torch.sum(exists_box * best_box_mask * (pred_x_y - target_x_y) ** 2)
    loss_w_h = torch.sum(exists_box * best_box_mask * (pred_w_h - target_w_h) ** 2)
    loss_coord = loss_x_y + loss_w_h

    # Podczas początkowych faz treningu pewność docelowa obiektu na starcie to 1.0 (zamiast max_iou)
    target_conf = max_iou.detach()
    # W późniejszych fazach można zmienić z powrotem na target_conf = max_iou.detach()
    loss_obj = torch.sum(exists_box * best_box_mask * (pred_conf - target_conf)**2)

    no_obj_mask = torch.ones_like(pred_conf) - (exists_box * best_box_mask)
    loss_noobj = torch.sum(no_obj_mask * (pred_conf - 0)**2)

    loss_class = torch.sum(exists_box.squeeze(3) * (pred_classes - target_classes)**2)

    # print(
    # loss_coord.item(),
    # loss_obj.item(),
    # loss_noobj.item(),
    # loss_class.item()
    # )

    loss_total = lambda_coord * loss_coord + loss_obj + lambda_noobj * loss_noobj + loss_class
    return loss_total / batch_size