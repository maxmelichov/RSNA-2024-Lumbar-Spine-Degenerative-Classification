import torch

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from model_code.network.vit_16_base_feat_middle_gal_v2 import vit_small_patch16_224, vit_base_patch16_256, vit_base_patch16_384
from torchvision.models import densenet121
import timm

class CustomRain(nn.Module):
    def __init__(self, num_classes=75, pretrained=True):
        super(CustomRain, self).__init__()
        self.cnv1 = nn.Conv2d(10, 30, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        self.cnv2 = nn.Conv2d(10, 30, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(30)
        self.cnv3 = nn.Conv2d(10, 30, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(30)
        self.cnv4 = nn.Conv2d(10, 30, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(30)
        self.cnv5 = nn.Conv2d(10, 30, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(30)
        self.saggital_l1_l2_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=60,
                                    num_classes=256,
                                    )
        self.saggital_l2_l3_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=60,
                                    num_classes=256,
                                    )
        self.saggital_l3_l4_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=60,
                                    num_classes=256,
                                    )
        self.saggital_l4_l5_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=60,
                                    num_classes=256,
                                    )
        self.saggital_l5_s1_backbone = timm.create_model(
                                    "timm/convnext_nano.in12k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=60,
                                    num_classes=256,
                                    )
        
        
        hdim1 = self.saggital_l1_l2_backbone.head.fc.in_features * 5
        self.saggital_l1_l2_backbone.head.fc = nn.Identity()
        self.saggital_l2_l3_backbone.head.fc = nn.Identity()
        self.saggital_l3_l4_backbone.head.fc = nn.Identity()
        self.saggital_l4_l5_backbone.head.fc = nn.Identity()
        self.saggital_l5_s1_backbone.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim1, 256, num_layers=2, dropout=0., bidirectional=True, batch_first=True)
        self.head1 = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(512, num_classes),
        )
        self.head2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 15),
        )
        self.head3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 15),
        )

        self.head4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 15),
        )

        self.head5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 15),
        )

    def forward(self, sagittal_l1_l2, sagittal_l2_l3, sagittal_l3_l4, sagittal_l4_l5,
                 sagittal_l5_s1, axial_l1_l2, axial_l2_l3, axial_l3_l4, axial_l4_l5, axial_l5_s1):
        axial_l1_l2 = self.cnv1(axial_l1_l2)
        axial_l1_l2 = self.bn1(axial_l1_l2)
        axial_l2_l3 = self.cnv2(axial_l2_l3)
        axial_l2_l3 = self.bn2(axial_l2_l3)
        axial_l3_l4 = self.cnv3(axial_l3_l4)
        axial_l3_l4 = self.bn3(axial_l3_l4)
        axial_l4_l5 = self.cnv4(axial_l4_l5)
        axial_l4_l5 = self.bn4(axial_l4_l5)
        axial_l5_s1 = self.cnv5(axial_l5_s1)
        axial_l5_s1 = self.bn5(axial_l5_s1)
        x1 = torch.cat([sagittal_l1_l2, axial_l1_l2], dim=1)
        x2 = torch.cat([sagittal_l2_l3, axial_l2_l3], dim=1)
        x3 = torch.cat([sagittal_l3_l4, axial_l3_l4], dim=1)
        x4 = torch.cat([sagittal_l4_l5, axial_l4_l5], dim=1)
        x5 = torch.cat([sagittal_l5_s1, axial_l5_s1], dim=1)
        x1 = x1.view(x1.size(0), x1.size(1), -1)
        x2 = x2.view(x2.size(0), x2.size(1), -1)
        x3 = x3.view(x3.size(0), x3.size(1), -1)
        x4 = x4.view(x4.size(0), x4.size(1), -1)
        x5 = x5.view(x5.size(0), x5.size(1), -1)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(x.size(0), 512*5)
        x1 = torch.cat([x, x1], dim=1)
        x2 = torch.cat([x, x2], dim=1)
        x3 = torch.cat([x, x3], dim=1)
        x4 = torch.cat([x, x4], dim=1)
        x5 = torch.cat([x, x5], dim=1)

        x1 = self.head1(x1)
        x2 = self.head2(x2)
        x3 = self.head3(x3)
        x4 = self.head4(x4)
        x5 = self.head5(x5)
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

    