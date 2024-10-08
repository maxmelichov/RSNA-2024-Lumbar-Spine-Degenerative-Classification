import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
from albumentations.pytorch import ToTensorV2  # Add this line to import ToTensorV2
from tqdm import tqdm

# Parameters
newsize = (512, 512)
fold = 1
batch_size = 32
num_workers = 4
num_classes = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
learning_rate = 1e-3
TRAIN = True  # Set to False for inference

im_dir = 'data_segmentation/images'
mask_dir = 'data_segmentation/masks'



# Dataset class
class SEGDataset(Dataset):
    def __init__(self, df, mode, transforms=None):
        self.df = df.reset_index()
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = os.path.join(im_dir, row.image)
        mask_path = os.path.join(mask_dir, row.image)

        # Open image
        image = Image.open(image_path)
        if image.mode != 'RGB':  # Ensure image is RGB
            image = image.convert('RGB')
        image = np.asarray(image)
        

        # Open mask
        mask = Image.open(mask_path)
        mask = np.asarray(mask)
        assert mask.max() < num_classes, f"Mask value {mask.max()} exceeds number of classes {num_classes}"

        image = np.copy(image)  # Ensure image array is writable
        mask = np.copy(mask)  # Ensure mask array is writable

        # Apply transformations
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Create one layer for each label
        mask = torch.as_tensor(mask).long()
        mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(2,0,1).float()
        #mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(0,3,1,2).squeeze(0).float()

        # Convert image to tensor
        image = torch.as_tensor(image).float()

        return image, mask  

# Transformations
def get_transforms(newsize):
    return A.Compose([
        A.Resize(newsize[0], newsize[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])

# Visualization function
def plot_segmentation_results(images, masks, predictions, num_classes=20):
    """
    Plots the images, true masks, and predicted masks.
    """
    num_images = min(4, len(images))  # Plot only 4 images
    fig, axs = plt.subplots(num_images, 3, figsize=(15, num_images * 5))

    for i in range(num_images):
        image = images[i].cpu().permute(1, 2, 0).numpy()
        true_mask = masks[i].cpu().argmax(0).numpy()
        predicted_mask = predictions[i].cpu().argmax(0).numpy()

        # Plot original image
        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Image")
        axs[i, 0].axis("off")

        # Plot true mask
        axs[i, 1].imshow(true_mask, cmap="gray", vmin=0, vmax=num_classes - 1)
        axs[i, 1].set_title("True Mask")
        axs[i, 1].axis("off")

        # Plot predicted mask
        axs[i, 2].imshow(predicted_mask, cmap="gray", vmin=0, vmax=num_classes - 1)
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

# Main function
if __name__ == '__main__':
    # Load your dataframe (assuming you have a DataFrame with image names)
    # Replace this with your actual dataframe loading method
    import pandas as pd
    df = pd.DataFrame({'image': os.listdir(im_dir)})

    # DataLoader
    train_dataset = SEGDataset(
        df=df,
        mode='train',
        transforms=get_transforms(newsize),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model Definition
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101", 
        encoder_weights="imagenet",  
        in_channels=3,  
        classes=num_classes,  # Multiclass segmentation
    )
    
    model_weight_path = "./weights_big_apex/deeplab_upernet_aspp_psp_ab_swin_skips_1288/deeplab_upe" \
                    "rnet_aspp_psp_ab_swin_skips_1288_0.0003.pth"
    new_state_dict = {}
    for key, value in torch.load(model_weight_path, map_location=torch.device('cpu')).items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    auto_cast = torch.cuda.amp.autocast(enabled=True, dtype = torch.bfloat16)
    best_loss = float('inf')
    # Training loop
    if TRAIN:
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            step = 0
            progress_bar = tqdm(train_loader, desc="Epoch 0/0", leave=False)
            for images, masks in progress_bar:
                with auto_cast:
                    images = images.to(device)
                    masks = masks.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)

                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                step += 1
                progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/step}")
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
            total_loss = running_loss/len(train_loader)
            if best_loss > total_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), f"segmentation.pt")
            # Visualize results after each epoch
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                sample_images, sample_masks = next(iter(train_loader))  # Get a batch of samples
                sample_images = sample_images.to(device)
                sample_predictions = model(sample_images)
                sample_predictions = sample_predictions.detach().cpu()  # Get predictions on CPU for visualization

            plot_segmentation_results(sample_images.cpu(), sample_masks.cpu(), sample_predictions, num_classes=num_classes)

        
