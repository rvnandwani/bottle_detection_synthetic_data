import torch
import cv2
import numpy as np
import os
import glob as glob
import json
from xml.etree import ElementTree as et
from config import classes, dir_train, ann_train, dir_val, ann_val, batch_size
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, dir_path, class_list, annFile, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.class_list = class_list
        self.all_image_paths = glob.glob(os.path.join(self.dir_path, "*.png"))
        self.all_images = [
            image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths
        ]
        self.all_images = sorted(self.all_images)
        with open(annFile, "r") as f:
            self.coco_format = json.load(f)
        self.all_images = self.coco_format["images"]
        self.anns = self.coco_format["annotations"]
        self.categories = self.coco_format["categories"]

    def __getitem__(self, idx):

        img_info = self.all_images[idx]
        img_path = os.path.join(self.dir_path, img_info["filename"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        ann_ids = [ann for ann in self.anns if ann["image_id"] == img_info["id"]]
        boxes = []
        labels = []
        for ann in ann_ids:
            boxes.append(ann["bbox"])
            labels.append(ann["categry_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_info["id"]])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image=img, bboxes=target["boxes"], labels=labels)
            img = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        if np.isnan((target["boxes"]).numpy()).any() or target[
            "boxes"
        ].shape == torch.Size([0]):
            target["boxes"] = torch.zeros((0, 4), dtype=torch.int64)
        return img, target

    def __len__(self):
        return len(self.all_images)


def create_train_dataset():
    train_dataset = CustomDataset(dir_train, classes, ann_train, get_train_transform())
    return train_dataset


def create_valid_dataset():
    valid_dataset = CustomDataset(dir_val, classes, ann_val, get_valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return valid_loader


if __name__ == "__main__":
    # Check for dataset pipeline
    dataset = CustomDataset(dir_train, classes, ann_train)
    print(f"Number of training images: {len(dataset)}")

    def visualize_sample(image, target, i):
        for box_num in range(len(target["boxes"])):
            box = target["boxes"][box_num]
            label = classes[target["labels"][box_num]]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 0),
                2,
            )
            cv2.putText(
                image,
                label,
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
        plt.imshow(image)
        plt.savefig(f"dataset_vis_{i}.png")

    for i in range(5):

        image, target = dataset[i]
        visualize_sample(image, target, i)
