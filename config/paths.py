import os

DATA_DIR = "coco/data"

IMG_DIR = os.path.join(DATA_DIR, "val2017")
ANN_DIR = os.path.join(DATA_DIR, "annotations")
ANN_FILE = os.path.join(ANN_DIR, "instances_val2017.json")

COCO_FILES = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}