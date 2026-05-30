import torch

from config.consts import C, S, B

def select_best_box(preds):

    original_shape = preds.shape[:-1]
    preds = preds.reshape(*original_shape, S, S, C + 5 * B)
    classes = preds[..., :C]
    boxes = preds[..., C:].reshape(*original_shape, S, S, B, 5)
    confidences = boxes[..., 0]

    best_idx = (
        confidences.argmax(dim=-1, keepdim=True)
        .unsqueeze(-1)
        .expand(*confidences.shape[:-1], 1, 5)
    )

    best_boxes = torch.gather(boxes, dim=-2, index=best_idx).squeeze(-2)
    output = torch.cat([classes, best_boxes], dim=-1)

    return output.reshape(*original_shape, S, S, (C + 5))
