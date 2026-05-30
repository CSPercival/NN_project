import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image

from config.consts import img_size

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform, img_size, S, B, C, cat_id_to_index):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.cat_id_to_index = cat_id_to_index
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transform:
            # if self.transform_mode == 0:
            #   image = self.transform(image)
            # else:
          image, target = self.transform(image, target, self.cat_id_to_index, (img_size, img_size), self.S, self.C)

        return image, target