device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

SEED = 23

img_size = 448

# grid size (number of cells)
S = 7

# bounding boxes per cell
B = 2

# number of classes
C = 80