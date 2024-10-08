import pandas as pd
import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
df_labels = pd.read_csv("train_label_coordinates.csv")
df_descriptions = pd.read_csv("train_series_descriptions.csv")

def create_rectangle(x, y, width, height):
    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2
    
    # Calculate top-left corner
    x1 = x - half_width
    y1 = y - half_height
    
    # Calculate bottom-right corner
    x2 = x + half_width
    y2 = y + half_height
    
    x = x1
    y = y1 - 5
    width = x2 - x1
    height = y2 - y1
    return (x, y, width, height)



import os
import pydicom
import pandas as pd
from tqdm import tqdm

# Path to DICOM images
train_path = "train_images"

# Create dataframe to store detection coordinates
detection_coordinates_df = pd.DataFrame(columns=["study_id", "series_id", "instance_number", "x", "y", "w", "h", "side"])

# Iterate through the label dataframe
for i in tqdm(range(len(df_labels))):
    study_id = df_labels.loc[i, "study_id"]
    series_id = df_labels.loc[i, "series_id"]
    instance_number = df_labels.loc[i, "instance_number"]
    
    # Get series description
    description = df_descriptions["series_description"].loc[
        (df_descriptions["study_id"] == study_id) & 
        (df_descriptions["series_id"] == series_id)
    ].iloc[0]
    
    if description == "Axial T2":
        x = df_labels.loc[i, "x"]
        y = df_labels.loc[i, "y"]
        
        # Read DICOM image
        image_path = os.path.join(train_path, str(study_id), str(series_id), f"{instance_number}.dcm")
        image = pydicom.dcmread(image_path).pixel_array
        
        # Create a bounding box based on the x, y coordinates
        w = image.shape[0] * 0.15  # Adjust as necessary for width
        h = image.shape[1] * 0.25  # Adjust as necessary for height

        # Determine side and adjust x-coordinate (Handle tilted images)
        if x < image.shape[0] // 2:
            x_adjusted = x - 60  # Move the bbox left by 40 units
            side = "left"
        else:
            x_adjusted = x + 10  # Move the bbox right by 10 units
            side = "right"
        
        bbox = (x_adjusted, y - 30, w, h)  # Adjust y-coordinate for better bbox positioning

        # Store detection coordinates in the dataframe
        detection_coordinates_df.loc[len(detection_coordinates_df)] = [study_id, series_id, instance_number, *bbox, side]

        # Check if the next instance is part of the same study and series
        if i < len(df_labels) - 1:  # Ensure we don't go out of bounds
            next_study_id = df_labels.loc[i + 1, "study_id"]
            next_series_id = df_labels.loc[i + 1, "series_id"]
            next_x = df_labels.loc[i + 1, "x"]
            next_y = df_labels.loc[i + 1, "y"]
            
            if study_id == next_study_id and series_id == next_series_id:
                # If current side is left and next side is right (or vice versa), calculate the center bbox
                if (x < image.shape[0] // 2 and next_x > image.shape[0] // 2) or (x > image.shape[0] // 2 and next_x < image.shape[0] // 2):
                    # Compute middle coordinates
                    center_x = image.shape[0] // 2  # Use center of the image
                    center_y = (y + next_y) // 2  # Middle of the y coordinates

                    # Define a new bbox for the center
                    center_bbox = (center_x, center_y - 30, w, h)

                    # Store the center bbox
                    detection_coordinates_df.loc[len(detection_coordinates_df)] = [study_id, series_id, instance_number, *center_bbox, "center"]

# Save detection coordinates to a CSV file
detection_coordinates_df.to_csv("detection_coordinates.csv", index=False)

    



import torch
from albumentations import Compose, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50
import pydicom
import pandas as pd
import os
import torchvision
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

class_mapping = {
    'left': 0,
    'right': 1,
    'center': 2
    # Add other classes as needed
}

class CustomDICOMDataset(Dataset):
    def __init__(self, root, csv_file, transforms=None):
        self.root = root
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        study_id = row['study_id']
        series_id = row['series_id']
        instance_number = row['instance_number']
        x, y, w, h = row['x'], row['y'], row['w'], row['h']
        side = row['side']  # Use `side` as the class label

        class_id = class_mapping.get(side, 0)  # Default to 0 if label is not found

        # Construct the DICOM file path
        dicom_path = os.path.join(self.root, str(study_id), str(series_id), f"{instance_number}.dcm")
        
        # Read DICOM image
        dicom_data = pydicom.dcmread(dicom_path)
        img = dicom_data.pixel_array
        original_height, original_width = img.shape
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)  # Convert to 3-channel image and ensure uint8 type
        

        scale_x = 256 / original_width
        scale_y = 256 / original_height
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        # Bounding box and label
        boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        labels = torch.tensor([class_id], dtype=torch.int64)
        
        # Assume all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = iscrowd
        
        if self.transforms:
            augmented = self.transforms(image=img)  # Pass as NumPy array
            img = augmented['image']
        
        return img, target, (original_height, original_width)

    def __len__(self):
        return len(self.data)

# Define transforms (normalize pixel values)
transforms = Compose([
    Resize(256, 256),  # Resize to a consistent size (H, W)
    HorizontalFlip(p=0.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()  # Use ToTensorV2 from Albumentations, which converts images to PyTorch tensors
])

# Create instances of the dataset
dataset = CustomDICOMDataset(root='train_images', csv_file='detection_coordinates.csv', transforms=transforms)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Define the model
backbone = resnet50(pretrained=True)
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 2048

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
    num_classes=4,  # Replace with number of classes in your dataset
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)

# Move model to the appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.00001, weight_decay=0.00001)

# Number of epochs
num_epochs = 20

# Training loop with tqdm
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_count = 0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    best_loss = 0
    for images, targets, _ in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backpropagation
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        losses.backward()
        optimizer.step()

        # Update running loss
        running_loss += losses.item()
        batch_count += 1
        progress_bar.set_postfix(loss=running_loss / batch_count)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss / len(data_loader)}")

    # Visualize model results on a random training sample
    model.eval()
    with torch.no_grad():
        sample_idx = random.randint(0, len(dataset) - 1)
        img, target, (original_height, original_width) = dataset[sample_idx]
        img = img.to(device)
        output = model([img])[0]

        # Plot the results
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
        img = np.clip(img, 0, 1)
        img = cv2.resize(img, (original_width, original_height))  # Resize image back to original size

        ax.imshow(img)
        
        # Rescale the bounding boxes to the original size
        rescale_x = original_width / 256
        rescale_y = original_height / 256

        # Plot ground truth boxes
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            xmin *= rescale_x
            xmax *= rescale_x
            ymin *= rescale_y
            ymax *= rescale_y
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Plot predicted boxes
        for box, score in zip(output['boxes'], output['scores']):
            if score > 0.5:  # Plot only high-confidence predictions
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                xmin *= rescale_x
                xmax *= rescale_x
                ymin *= rescale_y
                ymax *= rescale_y
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.title(f'Epoch {epoch+1}')
        plt.show()

    model.train()  # Set the model back to training mode
    total_loss = running_loss / len(data_loader)
    if best_loss < total_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), 'fasterrcnn_detection.pth')
    else:
        break
    # Save the model after training
    
