import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_coco(image_tensor, target, id_to_name):
    img = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    height, width, _ = img.shape

    for box, label in zip(target['boxes'], target['labels']):
      x1, y1, x2, y2 = box

      real_w = abs(x1 - x2)
      real_h = abs(y1 - y2)
      xmin = min(x1, x2)
      ymin = min(y1, y2)

      rect = patches.Rectangle(
          (xmin, ymin), real_w, real_h,
          linewidth=2, edgecolor='r', facecolor='none'
      )
      ax.add_patch(rect)

      label_name = id_to_name[label.item()]

      ax.text(
          xmin, ymin - 5,
          label_name,
          color='white',
          fontsize=10,
          bbox=dict(facecolor='red', alpha=0.5, pad=2)
      )

    plt.title("Normal Visualization")
    plt.axis('off')
    plt.show()


def visualize_yolo(image_tensor, yolo_target, S, C, id_to_name, conf_threshold=0.5):
    """
    image_tensor: (3, H, W)
    yolo_target: (S, S, C+5)
    """

    img = image_tensor.permute(1, 2, 0).numpy()
    H, W, _ = img.shape

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    cell_w = W / S
    cell_h = H / S

    for i in range(S):
        for j in range(S):
            cell = yolo_target[i, j]
            obj = cell[C]

            if obj < conf_threshold:
                continue

            x_rel, y_rel, w, h = cell[C+1:C+5]

            cx = (j + x_rel) * cell_w
            cy = (i + y_rel) * cell_h

            bw = w * W
            bh = h * H

            x1 = cx - bw / 2
            y1 = cy - bh / 2

            class_id = torch.argmax(cell[:C]).item()
            label = id_to_name[class_id] if id_to_name else str(class_id)

            rect = patches.Rectangle(
                (x1, y1), bw, bh,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x1, y1 - 5,
                label,
                color='white',
                fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5, pad=2)
            )

    plt.title("YOLO Grid Visualization")
    plt.axis('off')
    plt.show()