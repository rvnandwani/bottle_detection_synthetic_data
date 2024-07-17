import torch # version 1.12.1 used for testing
import torchvision # version 0.13.1 used for testing
import numpy as np
if __name__ == "__main__":

    # loads model with pretrained weights
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)

    # one way to update classification head only while keeping other weights still pretrained
    num_classes = 2
    size = (320, 320)
    num_anchors = model.anchor_generator.num_anchors_per_location() # output: [6, 6, 6, 6, 6, 6]
    out_channels = torchvision.models.detection._utils.retrieve_out_channels(model.backbone, size) # output: [672, 480, 512, 256, 256, 128]
    model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(out_channels, num_anchors, num_classes)

    # outputs model architecture to terminal
    print(model)

    # two random valued images
    x = [torch.rand(3, 512, 512), torch.rand(3, 512, 512)]

    # add fake labels
    targets = []
    for i in range(2):
        boxes = np.array([[100,100,200,200],[300,100,400,400]])
        labels = np.array([0,1])
        target = {}
        target["boxes"] = torch.from_numpy(boxes).type(torch.FloatTensor)
        target["labels"] = torch.from_numpy(labels).type(torch.LongTensor)
        targets.append(target)

    # get losses
    model.train()
    losses = model(x, targets)
    print("box regression loss:", losses["bbox_regression"])
    print("classification loss:", losses["classification"])

    # eval
    model.eval()
    with torch.no_grad():
        predictions = model(x)

    boxes = [det["boxes"].numpy() for det in predictions]
    scores = [det["scores"].numpy() for det in predictions]
    labels = [det["labels"].numpy() for det in predictions]
    print("labels:", labels)


