from config import device, class_num, epochs, res_dir, num_workers, dir_val, dir_train
from model import InbuiltSSD, ResnetWithAnchors
from utils import Averager, SaveBestModel, save_model, save_loss_plot, save_mAP
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset,
    create_valid_dataset,
    create_train_loader,
    create_valid_loader,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

import torch
import matplotlib.pyplot as plt
import time
import os


# Function for running training iterations.
def train(train_data_loader, model):
    print("Training")
    model.train()

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = torch.stack(list(image.to(device) for image in images)).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_track.send(loss_value)

        losses.backward()
        optimizer.step()

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value


# Function for running validation iterations.
def validate(valid_data_loader, model):
    print("Validating")
    model.eval()

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        # mAP calculation

        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict["boxes"] = targets[i]["boxes"].detach().cpu()
            true_dict["labels"] = targets[i]["labels"].detach().cpu()
            preds_dict["boxes"] = outputs[i]["boxes"].detach().cpu()
            preds_dict["scores"] = outputs[i]["scores"].detach().cpu()
            preds_dict["labels"] = outputs[i]["labels"].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary


if __name__ == "__main__":
    
    os.makedirs("outputs", exist_ok=True)
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, num_workers)
    valid_loader = create_valid_loader(valid_dataset, num_workers)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    seed = 15
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed Value set to : ", seed)

    #################################
    # Initialize the model
    # model = ResnetWithAnchors(num_classes=class_num)\
    model = InbuiltSSD(num_classes=class_num)
    model = model.to(device)
    print(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.1, verbose=True)
    print("Optimizer used is SGD and Parameters set as follows : ")
    print(
        " Learning Rate : 0.0001\n Momentum : 0.9\n Step size : 15\n Gamma : 0.1",
    )

    train_loss_track = Averager()
    train_loss_list = []
    map_50_list = []
    map_list = []

    save_best_model = SaveBestModel()

    # Training
    for epoch in range(epochs):
        print(f"\nEPOCH {epoch+1} of {epochs}")

        # Clear the training loss histories for the current epoch
        train_loss_track.reset()

        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        print(f"Epoch {epoch+1} train loss: {train_loss_track.value()}")
        print(f"Epoch {epoch+1} mAP: {metric_summary['map']}")
        end = time.time()
        print(f" {((end - start) / 60):.3f} minutes taken to complete epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary["map_50"])
        map_list.append(metric_summary["map"])

        # Save the best model so far
        save_best_model(model, float(metric_summary["map"]), epoch, "outputs")
        save_model(epoch, model, optimizer)

        # Saving Graphs
        save_loss_plot(res_dir, train_loss_list)
        save_mAP(res_dir, map_50_list, map_list)
        scheduler.step()
