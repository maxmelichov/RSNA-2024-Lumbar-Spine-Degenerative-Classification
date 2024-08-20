import torch

import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d
import timm

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='timm/efficientnet_b3.ra2_in1k', pretrained=True):
        super().__init__()
        # Load the pre-trained EfficientNet
        self.base_model = timm.create_model(model_name, pretrained=pretrained, in_chans=25)
        # Extract the feature extraction part and ignore the rest
        self.feature_extractor = nn.Sequential(
            *list(self.base_model.children())[:2]  # Assuming we want features before the final classifier
        )
    def forward(self, x):
        # Apply the layers up to the target layer
        x = self.feature_extractor(x)
        return x


class CustomRainFirstTry(nn.Module):
    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()
        self.axial_l1_l2_backbone = EfficientNetFeatureExtractor()
        self.axial_l2_l3_backbone = EfficientNetFeatureExtractor()
        self.axial_l3_l4_backbone = EfficientNetFeatureExtractor()
        self.axial_l4_l5_backbone = EfficientNetFeatureExtractor()
        self.axial_l5_s1_backbone = EfficientNetFeatureExtractor()
        self.saggital_l1_l2_backbone = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=40+15,
                                    num_classes=num_classes,
                                    )
        self.saggital_l2_l3_backbone = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=40+15,
                                    num_classes=num_classes,
                                    )
        self.saggital_l3_l4_backbone = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=40+15,
                                    num_classes=num_classes,
                                    )
        self.saggital_l4_l5_backbone = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=40+15,
                                    num_classes=num_classes,
                                    )
        self.saggital_l5_s1_backbone = timm.create_model(
                                    "timm/davit_small.msft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=40+15,
                                    num_classes=num_classes,
                                    )
        
        
        # hdim1 = self.saggital_l1_l2_backbone.head.fc.in_features * 5
        # self.saggital_l1_l2_backbone.head.fc = nn.Identity()
        # self.saggital_l2_l3_backbone.head.fc = nn.Identity()
        # self.saggital_l3_l4_backbone.head.fc = nn.Identity()
        # self.saggital_l4_l5_backbone.head.fc = nn.Identity()
        # self.saggital_l5_s1_backbone.head.fc = nn.Identity()

        # self.lstm = nn.LSTM(hdim1, 256, num_layers=2, dropout=0., bidirectional=True, batch_first=True)
        # self.head1 = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, num_classes),
        # )
        # self.head2 = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, 15),
        # )
        # self.head3 = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, 15),
        # )

        # self.head4 = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, 15),
        # )

        # self.head5 = nn.Sequential(
        #     nn.Linear(1152, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, 15),
        # )

    def forward(self, sagittal_l1_l2, sagittal_l2_l3, sagittal_l3_l4, sagittal_l4_l5,
                 sagittal_l5_s1, axial_l1_l2, axial_l2_l3, axial_l3_l4, axial_l4_l5, axial_l5_s1):
        axial_l1_l2 = self.axial_l1_l2_backbone(axial_l1_l2)
        axial_l2_l3 = self.axial_l2_l3_backbone(axial_l2_l3)
        axial_l3_l4 = self.axial_l3_l4_backbone(axial_l3_l4)
        axial_l4_l5 = self.axial_l4_l5_backbone(axial_l4_l5)
        axial_l5_s1 = self.axial_l5_s1_backbone(axial_l5_s1)
        x1 = torch.cat([sagittal_l1_l2, axial_l1_l2], dim=1)
        x2 = torch.cat([sagittal_l2_l3, axial_l2_l3], dim=1)
        x3 = torch.cat([sagittal_l3_l4, axial_l3_l4], dim=1)
        x4 = torch.cat([sagittal_l4_l5, axial_l4_l5], dim=1)
        x5 = torch.cat([sagittal_l5_s1, axial_l5_s1], dim=1)
        x1 = self.saggital_l1_l2_backbone(x1)
        x2 = self.saggital_l2_l3_backbone(x2)
        x3 = self.saggital_l3_l4_backbone(x3)
        x4 = self.saggital_l4_l5_backbone(x4)
        x5 = self.saggital_l5_s1_backbone(x5)
        # x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        # x = x.view(x.size(0), x.size(1) * 1)
        # x, _ = self.lstm(x)

        # batch_size = x.size(0)
        # num_features = x.size(1)
        # x = x.contiguous().view(batch_size, num_features)

        # x1 = x1.view(x1.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x3 = x3.view(x3.size(0), -1)
        # x4 = x4.view(x4.size(0), -1)
        # x5 = x5.view(x5.size(0), -1)
        
        # x1 = torch.cat([x, x1], dim=1)
        # x2 = torch.cat([x, x2], dim=1)
        # x3 = torch.cat([x, x3], dim=1)
        # x4 = torch.cat([x, x4], dim=1)
        # x5 = torch.cat([x, x5], dim=1)

        # x1 = self.head1(x1)
        # x2 = self.head2(x2)
        # x3 = self.head3(x3)
        # x4 = self.head4(x4)
        # x5 = self.head5(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return x
        x1, _ = self.lstm(x1)
        x2, _ = self.lstm2(x2)
        x3, _ = self.lstm3(x3)
        x4, _ = self.lstm4(x4)
        x5, _ = self.lstm5(x5)
        x1 = x1.contiguous().view(x1.size(0), 512)
        x2 = x2.contiguous().view(x2.size(0), 512)
        x3 = x3.contiguous().view(x3.size(0), 512)
        x4 = x4.contiguous().view(x4.size(0), 512)
        x5 = x5.contiguous().view(x5.size(0), 512)
        x1 = self.head(x1)
        x2 = self.head2(x2)
        x3 = self.head3(x3)
        x4 = self.head4(x4)
        x5 = self.head5(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return x


class CustomRain(nn.Module):
    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()
        # self.sagittal_stack_backbone = timm.create_model(
        #                             "timm/swin_base_patch4_window12_384.ms_in1k",
        #                             pretrained=pretrained, 
        #                             features_only=False,
        #                             in_chans=30,
        #                             num_classes = 0
        #                             )
        self.axial_l1_l2_backbone = timm.create_model(
                                    "timm/vit_small_patch14_reg4_dinov2.lvd142m",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=256,
                                    )
        self.axial_l2_l3_backbone = timm.create_model(
                                    "timm/vit_small_patch14_reg4_dinov2.lvd142m",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=256,
                                    )
        self.axial_l3_l4_backbone = timm.create_model(
                                    "timm/vit_small_patch14_reg4_dinov2.lvd142m",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=256,
                                    )
        self.axial_l4_l5_backbone = timm.create_model(
                                    "timm/vit_small_patch14_reg4_dinov2.lvd142m",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=256,
                                    )
        self.axial_l5_s1_backbone = timm.create_model(
                                    "timm/vit_small_patch14_reg4_dinov2.lvd142m",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=256,
                                    )
        self.sagittal_l1_l2_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=5,
                                    num_classes=256,
                                    )
        self.sagittal_l2_l3_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=5,
                                    num_classes=256
                                    )
        self.sagittal_l3_l4_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=5,
                                    num_classes=256,
                                    )
        self.sagittal_l4_l5_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=5,
                                    num_classes=256,
                                    )
        self.sagittal_l5_s1_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=5,
                                    num_classes=256,
                                    )

        self.l1_l2 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 15),
        )
        self.l2_l3 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 15),
        )
        self.l3_l4 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 15),
        )
        self.l4_l5 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 15),
        )
        self.l5_s1 = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, 15),
        )
        
 

        

    def forward(self, axial_l1_l2, sagittal_T2_l1_l2, axial_l2_l3,
                    sagittal_T2_l2_l3, axial_l3_l4, sagittal_T2_l3_l4, axial_l4_l5,
                      sagittal_T2_l4_l5,axial_l5_s1, sagittal_T2_l5_s1):
        # sagittal_stack = self.sagittal_stack_backbone(sagittal_stack)
        axial_l1_l2 = self.axial_l1_l2_backbone(axial_l1_l2)
        axial_l2_l3 = self.axial_l2_l3_backbone(axial_l2_l3)
        axial_l3_l4 = self.axial_l3_l4_backbone(axial_l3_l4)
        axial_l4_l5 = self.axial_l4_l5_backbone(axial_l4_l5)
        axial_l5_s1 = self.axial_l5_s1_backbone(axial_l5_s1)
        sagittal_T2_l1_l2 = self.sagittal_l1_l2_backbone(sagittal_T2_l1_l2)
        sagittal_T2_l2_l3 = self.sagittal_l2_l3_backbone(sagittal_T2_l2_l3)
        sagittal_T2_l3_l4 = self.sagittal_l3_l4_backbone(sagittal_T2_l3_l4)
        sagittal_T2_l4_l5 = self.sagittal_l4_l5_backbone(sagittal_T2_l4_l5)
        sagittal_T2_l5_s1 = self.sagittal_l5_s1_backbone(sagittal_T2_l5_s1)
        # print(sagittal_stack.shape, axial_l1_l2.shape, sagittal_T2_l1_l2.shape)
        x1 = torch.cat([sagittal_T2_l1_l2, axial_l1_l2], dim=1)
        x2 = torch.cat([sagittal_T2_l2_l3, axial_l2_l3], dim=1)
        x3 = torch.cat([sagittal_T2_l3_l4, axial_l3_l4], dim=1)
        x4 = torch.cat([sagittal_T2_l4_l5, axial_l4_l5], dim=1)
        x5 = torch.cat([sagittal_T2_l5_s1, axial_l5_s1], dim=1)
        x1 = self.l1_l2(x1)
        x2 = self.l2_l3(x2)
        x3 = self.l3_l4(x3)
        x4 = self.l4_l5(x4)
        x5 = self.l5_s1(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)


        return x
    