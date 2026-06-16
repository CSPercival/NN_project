import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

SEED = 23

img_size = 448

# grid size (number of cells)
S = 7

# bounding boxes per cell
B = 2

available_classes = ["bus", "train", "airplane","elephant", "zebra", "giraffe"]
# number of classes
C = len(available_classes)
# C = 80

lambda_coord = 5
lambda_noobj = 0.5

conf_threshold = 0.7
iou_threshold = 0.5
dropout_rate = 0.5

# batch_size = 64
batch_size = 16
# epochs = 135
epochs = 10
lr = 5e-5
weight_decay = 5e-4
