import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from config import device, classes


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    def __init__(self, best_valid_map=0.0):
        self.best_valid_map = best_valid_map

    def __call__(
        self,
        model,
        current_valid_map,
        epoch,
        res_dir,
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nValidation mAP: {self.best_valid_map}")
            print(f"\nBest model saved for: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                f"{res_dir}/best_model.pth",
            )


def collate_fn(batch):
    return tuple(zip(*batch))


# Define the training tranforms.
def get_train_transform():
    return A.Compose(
        [
            A.Blur(blur_limit=3, p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.ToGray(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.RandomGamma(p=0.3),
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


# Define the validation transforms.
def get_valid_transform():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )

def save_model(epoch, model, optimizer):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "outputs/last_model.pth",
    )


def save_loss_plot(
    res_dir,
    train_loss_list,
    x_label="iterations",
    y_label="train loss",
    save_name="train_loss",
):

    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color="tab:blue")
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{res_dir}/{save_name}.png")
    print("Plots saved...")


def save_mAP(res_dir, map_05, map):
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(map_05, color="tab:orange", linestyle="-", label="mAP@0.5")
    ax.plot(map, color="tab:red", linestyle="-", label="mAP@0.5:0.95")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("mAP")
    ax.legend()
    figure.savefig(f"{res_dir}/map.png")
