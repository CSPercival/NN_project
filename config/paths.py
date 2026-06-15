import os

DATA_DIR = "coco/data"

IMG_DIR_VAL = os.path.join(DATA_DIR, "val2017")
ANN_DIR_VAL = os.path.join(DATA_DIR, "annotations")
ANN_FILE_VAL = os.path.join(ANN_DIR_VAL, "instances_val2017.json")

COCO_FILES_VAL = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}


COCO_FILES_TRAIN = {
    "images": "http://images.cocodataset.org/zips/train2017.zip", # Zmiana tutaj
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}
IMG_DIR_TRAIN = os.path.join(DATA_DIR, "train2017") # I zmiana tutaj
ANN_DIR_TRAIN = os.path.join(DATA_DIR, "annotations")
ANN_FILE_TRAIN = os.path.join(ANN_DIR_TRAIN, "instances_train2017.json") # I zmiana tutaj