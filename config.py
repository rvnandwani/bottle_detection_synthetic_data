import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 10  # Batch size (Increase / decrease according to GPU memeory)

epochs = 35  # Number of epochs to train for.
num_workers = 4  # Number of parallel workers

# Training images and json annotation files directory
dir_train = "dataset/train/images"
ann_train = "dataset/train/annotations_train.json"
# Validation images and json annotation files directory
dir_val = "dataset/validation/images"
ann_val = "dataset/validation/annotations_validation.json"


classes = ["__background__", "bottle", "can"]

class_num = len(classes)

# Location to save model and plots.
res_dir = "outputs"
