import torch
from PIL import Image
import json
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annFile, 'r') as f:
            self.coco = json.load(f)
        self.imgs = self.coco['images']
        self.anns = self.coco['annotations']
        self.categories = self.coco['categories']

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = os.path.join(self.root, img_info['filename'])
        img = Image.open(img_path).convert("RGB")
        ann_ids = [ann for ann in self.anns if ann['image_id'] == img_info['id']]
        boxes = []
        labels = []
        for ann in ann_ids:
            boxes.append(ann['bbox'])
            labels.append(ann['categry_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_info['id']])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)