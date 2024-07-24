import torch

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from network.vit_16_base_feat_middle_gal_v2 import vit_base_patch16_384


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.EfficientNet_model_Sagittal_T1 = EfficientNet.from_pretrained('efficientnet-b5')
        self.EfficientNet_model_Axial_T2 = EfficientNet.from_pretrained('efficientnet-b5')
        self.EfficientNet_model_Sagittal_T2_STIR = EfficientNet.from_pretrained('efficientnet-b5')
        self.transformer = vit_base_patch16_384()

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR):
        '''
        Sagittal1 - Sagittal1 side view that has a connetion with Axial view
        Axial T2 - the view from the top
        Sagittal T2 STIR - Axial view that has a connection with Sagittal1
        '''
        Sagittal_T1 = self.EfficientNet_model_Sagittal_T1.extract_endpoints(Sagittal_T1)
        Sagittal_T1 = Sagittal_T1['reduction_2']
        Axial_T2 = self.EfficientNet_model_Axial_T2.extract_endpoints(Axial_T2)
        Axial_T2 = Axial_T2['reduction_2']
        Sagittal_T2_STIR = self.EfficientNet_model_Sagittal_T2_STIR.extract_endpoints(Sagittal_T2_STIR)
        Sagittal_T2_STIR = Sagittal_T2_STIR['reduction_2']
        Concatinate = torch.cat((Sagittal_T1, Axial_T2, Sagittal_T2_STIR), 1)
        x = self.transformer(Concatinate)
        return x


# Create an instance of your model
model = CustomModel()

# Print the model architecture
print(model)