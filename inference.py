import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

from model import InbuiltSSD, ResnetWithAnchors

from config import device, class_num, classes


# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    help="path to input image directory",
)
parser.add_argument(
    "-o",
    "--output",
    default="inference_outputs/images",
    help="path to result image directory",
)
parser.add_argument("--threshold", default=0.25, type=float, help="detection threshold")
args = vars(parser.parse_args())

out_dir = args["output"]
os.makedirs(out_dir, exist_ok=True)

colors = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

#################################
# Load the best model and trained weights.
model = ResnetWithAnchors(num_classes=class_num)
# model = InbuiltSSD(num_classes=class_num)

checkpoint = torch.load("outputs/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Directory where all the images are present.
input_dir = args["input"]
test_images = glob.glob(f"{input_dir}/*.png")
print(f"Test instances: {len(test_images)}")

for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split(".")[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).cuda()
    image_input = torch.unsqueeze(image_input, 0)
    with torch.no_grad():
        outputs = model(image_input.to(device))

    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
    if len(outputs[0]["boxes"]) != 0:
        boxes = outputs[0]["boxes"].data.numpy()
        scores = outputs[0]["scores"].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= args["threshold"]].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [classes[i] for i in outputs[0]["labels"].cpu().numpy()]

        # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            # Recale boxes.
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color[::-1], 3)
            cv2.putText(
                orig_image,
                class_name,
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color[::-1],
                2,
                lineType=cv2.LINE_AA,
            )

        cv2.imwrite(f"{out_dir}/target_ssd_e50_{image_name}.png", orig_image)
    print(f"Image {i+1} done...")
    print("-" * 50)

print("Inference complete !!")
cv2.destroyAllWindows()
