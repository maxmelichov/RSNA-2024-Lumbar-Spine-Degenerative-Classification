import torch

import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define your layers here

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR):
        '''
        Sagittal1 - Sagittal1 side view that has a connetion with Axial view
        Axial T2 - the view from the top
        Sagittal T2 STIR - Axial view that has a connection with Sagittal1
        '''        
        pass

# Create an instance of your model
model = CustomModel()

# Print the model architecture
print(model)