import torch

from config.consts import C, S, B

def select_best_box(preds):

    original_shape = preds.shape[:-1]
    preds = preds.reshape(*original_shape, S, S, C + 5 * B)
    classes = torch.sigmoid(preds[..., :C])
    
    boxes = preds[..., C:].reshape(*original_shape, S, S, B, 5)
    
    # Apply sigmoids and absolute values just like in target format extraction
    confidences = torch.sigmoid(boxes[..., 0:1])
    x_y = torch.sigmoid(boxes[..., 1:3])
    w_h = torch.abs(boxes[..., 3:5]) # We enforce positive width/height
    
    processed_boxes = torch.cat([confidences, x_y, w_h], dim=-1)

    confidences_dist = processed_boxes[..., 0]

    best_idx = (
        confidences_dist.argmax(dim=-1, keepdim=True)
        .unsqueeze(-1)
        .expand(*confidences_dist.shape[:-1], 1, 5)
    )

    best_boxes = torch.gather(processed_boxes, dim=-2, index=best_idx).squeeze(-2)
    output = torch.cat([classes, best_boxes], dim=-1)

    return output.reshape(*original_shape, S, S, (C + 5))
