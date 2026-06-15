import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image

from config.consts import img_size

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform, img_size, S, B, C, categories=None, filter_flag=False):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        self.filter_dataset(categories, filter_flag)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item["image_id"]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        anns = item["anns"]

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
            image, target = self.transform(image, target, (img_size, img_size), self.S, self.C)

        return image, target
    
    def filter_dataset(self, new_categories, filter_flag):
        if(filter_flag == False):
            cats = self.coco.loadCats(self.coco.getCatIds())
            new_categories = [cat['name'] for cat in cats]

        self.data = []

        old_categories = self.coco.loadCats(self.coco.getCatIds())
        old_id_to_name = {cat['id']: cat['name'] for cat in old_categories}
        old_name_to_id = {cat['name']: cat['id'] for cat in old_categories}
        id_old_to_new = {}
        id_new_to_old = {}

        for nc in new_categories:
            if nc not in old_name_to_id:
                raise ValueError(f"Category '{nc}' not found in COCO dataset")
            else:
                old_id = old_name_to_id[nc]
                new_id = len(id_old_to_new)
                id_old_to_new[old_id] = new_id
                id_new_to_old[new_id] = old_id
        
        self.id_to_category_name = new_categories
        self.category_name_to_id = {name: id for id, name in enumerate(new_categories)}

        # Słownik do śledzenia liczby obrazków dla każdej wybranej klasy
        class_image_counts = {name: 0 for name in new_categories}

        for image_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            filtered_anns = []
            present_classes_in_image = set()

            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in id_old_to_new:
                    continue
                
                new_cat_id = id_old_to_new[cat_id]
                ann["category_id"] = new_cat_id
                filtered_anns.append(ann)
                
                # Dodajemy nazwę klasy do zbioru, aby zapisać, że występuje na tym zdjęciu
                present_classes_in_image.add(self.id_to_category_name[new_cat_id])

            if len(filtered_anns) == 0:
                continue
            
            # Zwiększamy licznik obrazków dla każdej unikalnej klasy wykrytej na tym zdjęciu
            for class_name in present_classes_in_image:
                class_image_counts[class_name] += 1

            self.data.append({
                "image_id": image_id,
                "anns": filtered_anns,
            })

        # Wypisywanie statystyk na konsolę po zakończeniu filtrowania
        print(f"\n--> [DATASET INFO] Statystyki po wyfiltrowaniu:")
        for class_name, count in class_image_counts.items():
            print(f"    - Klasa '{class_name}' występuje na {count} obrazkach.")
        print(f"    [ŁĄCZNIE]: Przefiltrowany zbiór zawiera {len(self.data)} unikalnych obrazków.\n")