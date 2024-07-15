
# INSTRUCTIONS
# ------------ 
# - For creating the object detector you need to make your OWN object detector class.
# - The model should have an backbone/feature extractor, neck (optional) & detector heads (e.g. classification, bbox and/or segmentation mask).
# - Each of the above mentioned parts need to be their separate method of the class, thought you can use "prebuild" parts taken from other repositories as mentioned on the README.md
# - The object detector class needs to be intialized by passing in parameters for the __init__ function.
# - The object detector class needs to have a 'forward' function that calls each of the model parts needed for forward pass.

# Below is an example of the general structure of the object detector

from torch import nn

class ObjectDetectionModel(nn.Module):
    def __init__(self, params):
        """
        model parts are initialized from the given parameters
        """
        super().__init__()
        self.backbone = self.prepare_backbone(params)
        self.neck = self.prepare_neck(params)
        self.cls_head, self.bbox_head = self.prepare_heads(params)

    def forward(self, x):
        """
        forward pass
        """
        h = self.backbone(x)
        h = self.neck(h)
        cls_preds = self.cls_head(h)
        bbox_preds = self.bbox_head(h)

        return cls_preds, bbox_preds

    def prepare_backbone(self, params):
        pass

    def prepare_neck(self, params):
        pass

    def prepare_heads(self, params):
        pass