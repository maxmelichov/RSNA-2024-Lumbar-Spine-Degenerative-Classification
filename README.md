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
results: 
fold 1:
Train Loss: 
Validation Loss:  
Last LR [0.00036]
Kaggle score: 


MUST ADD LABEL ORDER TO INFERENCE!