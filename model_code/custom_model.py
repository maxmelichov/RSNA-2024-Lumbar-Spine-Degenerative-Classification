import torch

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from model_code.network.vit_16_base_feat_middle_gal_v2 import vit_base_patch16_384, vit_base_patch16_256
from torchvision.models import densenet121
import timm

class CustomModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomModel, self).__init__()
        self.EfficientNet_model_Sagittal_T1 = EfficientNet.from_pretrained('efficientnet-b5', in_channels= 15)
        self.EfficientNet_model_Axial_T2 = EfficientNet.from_pretrained('efficientnet-b5', in_channels= 10)
        self.EfficientNet_model_Sagittal_T2_STIR = EfficientNet.from_pretrained('efficientnet-b5' , in_channels= 15)
        self.transformer = vit_base_patch16_256(pretrained=False, num_classes=num_classes)

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR, **kwargs):
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
        x = self.transformer(Concatinate, **kwargs)
        return x


class CustomModel2(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomModel2, self).__init__()
        self.EfficientNet_model_Sagittal_T1 = EfficientNet.from_pretrained('efficientnet-b5', in_channels= 3)
        self.EfficientNet_model_Axial_T2 = EfficientNet.from_pretrained('efficientnet-b5', in_channels= 10)
        self.EfficientNet_model_Sagittal_T2_STIR = EfficientNet.from_pretrained('efficientnet-b5' , in_channels= 3)
        self.densenet121 = timm.create_model(
                                    "densenet121",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=120,
                                    num_classes=251,
                                    global_pool='avg'
                                    )
        self.fc1 = nn.Linear(251 + 5, 128)  # First fully connected layer
        self.relu = nn.SELU()               # Activation function
        self.fc2 = nn.Linear(128, num_classes)  # Second fully connected layer

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR, category_hot):
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
        x = self.densenet121(Concatinate)
        x = torch.cat((x, category_hot), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomModel3(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomModel3, self).__init__()
        self.EfficientNet_model_Sagittal_T1 = EfficientNet.from_pretrained('efficientnet-b3', in_channels= 15)
        self.EfficientNet_model_Axial_T2 = EfficientNet.from_pretrained('efficientnet-b3', in_channels= 10)
        self.EfficientNet_model_Sagittal_T2_STIR = EfficientNet.from_pretrained('efficientnet-b3' , in_channels= 15)
        self.EfficientNet = EfficientNet.from_pretrained("efficientnet-b7", in_channels= 96)
        self.number_classes = num_classes
        self.liner = nn.Linear(1000 + 5, self.number_classes)

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR, category_hot):
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
        x = self.EfficientNet(Concatinate)
        x = torch.cat((x, category_hot), 1)
        x = self.liner(x)
        return x
    

class CustomModel4(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomModel4, self).__init__()
        self.davit_model_Sagittal_T1 = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=3)
        self.davit_model_Axial_T2 = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=10)
        self.davit_model_Sagittal_T2_STIR = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=3)
        self.densenet121 = timm.create_model(
                                    "densenet121",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=120,
                                    num_classes=251,
                                    global_pool='avg'
                                    )
        self.fc1 = nn.Linear(251 + 5, 128)  # First fully connected layer
        self.relu = nn.SELU()               # Activation function
        self.fc2 = nn.Linear(128, num_classes)  # Second fully connected layer

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR, category_hot):
        '''
        Sagittal1 - Sagittal1 side view that has a connetion with Axial view
        Axial T2 - the view from the top
        Sagittal T2 STIR - Axial view that has a connection with Sagittal1
        '''
        Sagittal_T1 = self.davit_model_Sagittal_T1.extract_endpoints(Sagittal_T1)
        Sagittal_T1 = Sagittal_T1['reduction_2']
        Axial_T2 = self.davit_model_Axial_T2.extract_endpoints(Axial_T2)
        Axial_T2 = Axial_T2['reduction_2']
        Sagittal_T2_STIR = self.davit_model_Sagittal_T2_STIR.extract_endpoints(Sagittal_T2_STIR)
        Sagittal_T2_STIR = Sagittal_T2_STIR['reduction_2']
        Concatinate = torch.cat((Sagittal_T1, Axial_T2, Sagittal_T2_STIR), 1)
        x = self.densenet121(Concatinate)
        x = torch.cat((x, category_hot), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class CustomModel5(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomModel5, self).__init__()
        self.EfficientNet_model_Sagittal_T1 = EfficientNet.from_pretrained('efficientnet-b3', in_channels= 3)
        self.EfficientNet_model_Axial_T2 = EfficientNet.from_pretrained('efficientnet-b3', in_channels= 10)
        self.EfficientNet_model_Sagittal_T2_STIR = EfficientNet.from_pretrained('efficientnet-b3' , in_channels= 3)
        self.densenet121 = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=96,
                                    num_classes=251,
                                    )
        self.fc1 = nn.Linear(251 + 5, 128)  # First fully connected layer
        self.relu = nn.SELU()               # Activation function
        self.fc2 = nn.Linear(128, num_classes)  # Second fully connected layer

    def forward(self, Sagittal_T1, Axial_T2, Sagittal_T2_STIR, category_hot):
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
        x = self.densenet121(Concatinate)
        x = torch.cat((x, category_hot), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class CustomModel6(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomModel6, self).__init__()
        self.densenet121 = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=True, 
                                    features_only=False,
                                    in_chans=30,
                                    num_classes=251,
                                    )
        self.fc1 = nn.Linear(251 + 5, 128)  # First fully connected layer
        self.relu = nn.SELU()               # Activation function
        self.fc2 = nn.Linear(128, num_classes)  # Second fully connected layer

    def forward(self, x, category_hot):
        '''
        Sagittal1 - Sagittal1 side view that has a connetion with Axial view
        Axial T2 - the view from the top
        Sagittal T2 STIR - Axial view that has a connection with Sagittal1
        '''
        x = self.densenet121(x)
        x = torch.cat((x, category_hot), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
# # Create an instance of your model

# model = CustomModel()

# # Print the model architecture
# print(model)