# RSNA-2024-Lumbar-Spine-Degenerative-Classification

What we need to do: 
Model output is 3 multilabel classes.
[class balance]("images_for_readme\class_balance.png")

you have 26 spots with the problem of the classification
[spot 1-3]("images_for_readme\1-3.png")
[spot 4-6]("images_for_readme\4-6.png")


# Trial: 
## Trial 1:
using davit_small with hot_category with Sagittal_T1, Sagittal_T2, Axial_T2 on top of each other
image size 512, 512, 30
crossentropy with weights
results: 
epoch 1:
Validation Loss:  0.8530751908979108
Last LR [0.00036]
Train Loss: 0.8827
