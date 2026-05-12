import torch

def YOLO_loss(preds, targets):

    preds = preds.reshape(-1, S, S, C + B * 5)
    batch_size = preds.size(0)

    pred_classes = preds[..., :C]
    pred_boxes = preds[..., C:].reshape(-1, S, S, B, 5)

    target_box = targets[..., C+1:C+5].unsqueeze(3)
    target_classes = targets[..., :C]
    exists_box = targets[..., C:C+1].unsqueeze(3)

    ious = iou(pred_boxes[..., 1:5], target_box)

    max_iou, best_box_id = torch.max(ious, dim=3, keepdim=True)
    best_box_mask = torch.zeros_like(ious).scatter_(3, best_box_id, 1)

    assigned_pred_boxes = best_box_mask * pred_boxes[..., 1:5]
    assigned_target_boxes = best_box_mask * target_box

    pred_w_h = assigned_pred_boxes[..., 2:4]
    target_w_h = assigned_target_boxes[..., 2:4]

    pred_w_h_sqrt = torch.sign(pred_w_h) * torch.sqrt(torch.abs(pred_w_h).clamp(1e-6))
    target_w_h_sqrt = torch.sqrt(target_w_h)

    final_coords_pred = torch.cat([assigned_pred_boxes[..., 0:2], pred_w_h_sqrt], dim=-1)
    final_coords_target = torch.cat([assigned_target_boxes[..., 0:2], target_w_h_sqrt], dim=-1)

    loss_coord = torch.sum(exists_box * (final_coords_pred - final_coords_target)**2)

    best_box_conf = best_box_mask * pred_boxes[..., 0:1]
    # target_conf = best_box_mask * max_iou
    # # maybe it's better to use 1? sources say the original iou isn't the greatest
    # loss_obj = torch.sum(exists_box * (best_box_conf - target_conf)**2)
    loss_obj = torch.sum(exists_box * best_box_mask * (best_box_conf - 1)**2)

    # no_obj_mask = ~ (exists_box.bool() & best_box_mask.bool())
    no_obj_mask = torch.ones_like(pred_boxes[..., 0:1])
    no_obj_mask = no_obj_mask - exists_box * best_box_mask
    loss_noobj = torch.sum(no_obj_mask * (pred_boxes[..., 0:1] - 0)**2)

    loss_class = torch.sum(exists_box.squeeze(3) * (pred_classes - target_classes)**2)

    return (lambda_coord * loss_coord + loss_obj + lambda_noobj * loss_noobj + loss_class) / batch_size