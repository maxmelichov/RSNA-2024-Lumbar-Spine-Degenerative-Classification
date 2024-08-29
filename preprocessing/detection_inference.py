import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18
import torchvision
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
class_mapping = {
    'left': 0,
    'right': 1,
    # Add other classes as needed
}

transforms = Compose([
    Resize(IMG_SIZE, IMG_SIZE),  # Resize to a consistent size (H, W)
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()  # Use ToTensorV2 from Albumentations, which converts images to PyTorch tensors
])

class DetectionInference:

    def __init__(self, model_path, transforms):
        self.model_path = model_path
        self.transforms = transforms
        self.load_model()

    # Load the model
    def load_model(self):
        # Define the model
        backbone = resnet18(pretrained=False)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 512

        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        model = FasterRCNN(
            backbone,
            num_classes=2,  # Replace with number of classes in your dataset
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler
        )
        model.load_state_dict(torch.load(self.model_path))
        self.model = model.to(device)
    def inference(self, img, original_height = None, original_width = None):
        if original_height is None or original_width is None:
            original_height, original_width = img.shape
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)
        if self.transforms:
            augmented = self.transforms(image=img)  # Pass as NumPy array
            img = augmented['image']
        
        self.model.eval()
        with torch.no_grad():
            output = self.model([img.to(device)])[0]
        # Rescale the bounding boxes to the original size
        rescale_x = original_width / IMG_SIZE
        rescale_y = original_height / IMG_SIZE

        x, y, w, h = [], [], [], []
        # Plot ground truth boxes
        for box, score in zip(output['boxes'], output['scores']):
            if score > 0.5:  # Plot only high-confidence predictions
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                xmin *= rescale_x
                xmax *= rescale_x
                ymin *= rescale_y
                ymax *= rescale_y
                x.append(xmin)
                y.append(ymin)
                w.append(xmax - xmin)
                h.append(ymax - ymin)
        bbox = list(zip(x, y, w, h))
        return bbox
