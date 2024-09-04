import torch

import torch.nn as nn
import timm

class CustomRain(nn.Module):
    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()
        self.stack_left_right_backbone = timm.create_model(
                                    "timm/edgenext_base.in21k_ft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=23,
                                    num_classes=6,
                                    )
        self.stack_center_backbone = timm.create_model(
                                    "timm/edgenext_base.in21k_ft_in1k",
                                    pretrained=pretrained,
                                    features_only=False,
                                    in_chans=23,
                                    num_classes=3,
                                    )

    def forward(self, stack):
        left = torch.cat((stack[:, :20, :, :], stack[:, 20:23, :, :]), dim=1)
        center = torch.cat((stack[:, :20, :, :], stack[:, 23:26, :, :]), dim=1)
        right = torch.cat((stack[:, :20, :, :], stack[:, 26:29, :, :]), dim=1)
        x_left = self.stack_left_right_backbone(left)
        x_center = self.stack_center_backbone(center)
        x_right = self.stack_left_right_backbone(right)
        x = torch.cat([x_center, x_left[:,:3], x_right[:,:3], x_left[:,3:], x_right[:,3:]], dim=1)
        return x
    