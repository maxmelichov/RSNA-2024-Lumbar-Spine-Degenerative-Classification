# RSNA-2024-Lumbar-Spine-Degenerative-Classification

What we need to do: 
Model output is 3 multilabel classes.
[class balance]("images_for_readme\class_balance.png")

you have 26 spots with the problem of the classification
[spot 1-3]("images_for_readme\1-3.png")
[spot 4-6]("images_for_readme\4-6.png")


# Trials: 
## Trial 1:
using davit_small with hot_category with Sagittal_T1, Sagittal_T2, Axial_T2 on top of each other
image size 384x384x10 for Axial_T2 and 192x192x15 for Sagittal_T1/T2
CustomLoss severeloss 
using timm/davit_small.msft_in1k net
number of epochs: 5
results: 
fold 1:
Train Loss: 0.802
Validation Loss:  0.806
Last LR [0.000036]
Kaggle score: 1.08

## Trial 2:
using davit_small with hot_category with Sagittal_T1, Sagittal_T2, Axial_T2 on top of each other
image size 384x384x10 for Axial_T2 and 192x192x15 for Sagittal_T1/T2
crossentropy with weights
using timm/davit_small.msft_in1k net
number of epochs: 5
results: 
fold 1:
Train Loss: 0.792 
Validation Loss:  0.728
Last LR [0.00036]
Kaggle score: 1.03

## Trial 3:
using davit_small with hot_category with Sagittal_T1, Sagittal_T2, Axial_T2 on top of each other
image size 384x384x10 for Axial_T2 384x384x15 for Sagittal_T1 and 192x192x15 for Sagittal_T2
crossentropy with weights
using timm/davit_small.msft_in1k net and for feature extactor timm/efficientnet_b3.ra2_in1k
number of epochs: 5
results: 
fold 1:
Train Loss: 0.76
Validation Loss: 0.79
Last LR [0.00036]
Kaggle score: 0.98


## Trial 4:
using davit_small with hot_category with Sagittal_T1, Sagittal_T2, Axial_T2 on top of each other
image size 384x384x10 for Axial_T2 384x384x15 for Sagittal_T1 and 192x192x15 for Sagittal_T2
crossentropy with weights
using timm/davit_small.msft_in1k net and for feature extactor timm/efficientnet_b3.ra2_in1k
using 5 models for each part
number of epochs: 5
results: 
fold 1:
Train Loss: 0.749
Validation Loss: 0.585
Last LR [0.00036]
Kaggle score: 0.95

## Trial 4:
using davit_small with hot_category with Sagittal_T1, Sagittal_T2, Axial_T2 on top of each other
image size 384x384x10 for Axial_T2 384x384x15 for Sagittal_T1 and 192x192x15 for Sagittal_T2
crossentropy with weights
using timm/davit_small.msft_in1k net and for feature extactor timm/efficientnet_b3.ra2_in1k
using 5 models for each part
number of epochs: 10
results: 
fold 1: after 4 fold the results on kaggle are the same
Train Loss: 0.749
Validation Loss: 0.585
Last LR [0.00036]
Kaggle score: 0.95


## Trial 5:
#### Summary:
The model is designed to take in axial and sagittal MRI scans of different sections of the lumbar spine, extract features using powerful pre-trained deep learning models (ConvNeXt for sagittal and axial), and then combine these features to predict some region-specific outcomes. The use of separate backbones for different regions and the combination of axial and sagittal features makes the model highly specialized for analyzing lumbar spine MRI data.

image size 512x512x3 for Axial_T2 128x128x5 for Sagittal_T2
crossentropy with weights
using timm/convnext_nano.in12k
number of epochs: 5
results: 
fold 1: after 4 fold the results on kaggle are the same
Train Loss: 0.6
Validation Loss: 0.7
Last LR [0.00036]
Kaggle score: 0.8

## Trial 5:
#### Summary:

The model is designed to take in axial and sagittal MRI scans of different sections of the lumbar spine, extract features using powerful pre-trained deep learning models (ViT for axial and ConvNeXt for sagittal), and then combine these features to predict some region-specific outcomes. The use of separate backbones for different regions and the combination of axial and sagittal features makes the model highly specialized for analyzing lumbar spine MRI data.

image size 518x518x3 for Axial_T2 128x128x5 for Sagittal_T2 not using Sagittal_T1
crossentropy with weights
using timm/convnext_nano.in12k for Sagittal_T2 and timm/vit_small_patch14_reg4_dinov2.lvd142m and for Axial_T2
number of epochs: 5
results: 
fold 2: after 4 fold the results on kaggle the result is much worst 1.36
Train Loss: 0.48
Validation Loss: 0.47
Last LR [0.00036]
Kaggle score: 0.64 for fold 1 I got 0.62
