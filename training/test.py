from data_loader import CustomDataset
from transform import get_transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_image(img, target, category_names):

    img = np.asarray(img)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img)
    for box, label in zip(target['boxes'], target['labels']):
        xmin, ymin, xmax, ymax = box
        width = xmax-xmin
        height = ymax-ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, category_names[label.item()], fontsize=15, color='white', bbox=dict(facecolor='red', alpha=0.5))
    output_path = 'visualization_test.png'
    plt.savefig(output_path)
    plt.close(fig)


root = 'dataset/train/images'
annFile = 'dataset/train/full_ann.json'
dataset = CustomDataset(root=root, annFile=annFile, transforms=get_transform(train=True))


category_names = {category['id']: category['name'] for category in dataset.categories}

img, target = dataset[0]
visualize_image(img, target, category_names)