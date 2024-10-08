import torch

import torch.nn as nn
import timm


class CustomRain(nn.Module):
    def __init__(self, num_classes=75, pretrained=True):
        super().__init__()
<<<<<<< HEAD
<<<<<<< HEAD
        self.stack_t1 = timm.create_model(
                                    "timm/edgenext_base.in21k_ft_in1k",
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=128,
                                    )

        self.stack_t2 = timm.create_model(
                                    "timm/edgenext_base.in21k_ft_in1k",
                                    pretrained=pretrained,
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=128,
                                    )
        self.stack_axial = timm.create_model(
                                    "timm/edgenext_base.in21k_ft_in1k",
                                    pretrained=pretrained,
                                    features_only=False,
                                    in_chans=3,
                                    num_classes=128,
                                    )
        self.stack = timm.create_model(
                                    "timm/edgenext_base.in21k_ft_in1k",
                                    pretrained=pretrained,
                                    features_only=False,
                                    in_chans=30,
                                    num_classes=128,)

        
        self.head_l = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(),
        )
        self.head_r = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(),
        )
        self.head_t2 = nn.Sequential(
            nn.Linear(128*3, 128),
            nn.ReLU(),
        )

        self.scs = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.nfn_l = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.nfn_r = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.ss_left = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.ss_right = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        # self.ascension_callback = AscensionCallback(margin=0.15)
        self.level_embeddings = nn.Embedding(5, 128)

    def forward(self, stack, level_idx):
        t1_l1 = stack[:, :3, :, :]
        t1_l2 = stack[:, 3:6, :, :]
        t1_r1 = stack[:, 6:9, :, :]
        t1_r2 = stack[:, 9:12, :, :]

        t2_1 = stack[:, 12:15, :, :]
        t2_2 = stack[:, 15:18, :, :]
        t2_3 = stack[:, 18:21, :, :]

        axial_left = stack[:, 21:24, :, :]
        axial_center = stack[:, 24:27, :, :]
        axial_right = stack[:, 27:, :, :]
        

        x_t1_l1 = self.stack_t1(t1_l1)
        x_t1_l2= self.stack_t1(t1_l2)

        x_t1_r1 = self.stack_t1(t1_r1)
        x_t1_r2 = self.stack_t1(t1_r2)

        x_t1_l = torch.cat((x_t1_l1, x_t1_l2), dim=1)
        x_t1_l = self.head_l(x_t1_l)

        x_t1_r = torch.cat((x_t1_r1, x_t1_r2), dim=1)
        x_t1_r = self.head_r(x_t1_r)

        x_t2_1 = self.stack_t2(t2_1)
        x_t2_2 = self.stack_t2(t2_2)
        x_t2_3 = self.stack_t2(t2_3)

        x_t2 = torch.cat((x_t2_1, x_t2_2, x_t2_3), dim=1)
        x_t2 = self.head_t2(x_t2)

        x_axial_left = self.stack_axial(axial_left)
        x_axial_center = self.stack_axial(axial_center)
        x_axial_right = self.stack_axial(axial_right)
        # stack[:, :21, :, :] = torch.rot90(stack[:, :21, :, :], k=3, dims=[2, 3])
        x_stack = self.stack(stack)
        bs = x_t2.size(0)
        level_embeddings = self.level_embeddings(level_idx)
        level_embeddings = level_embeddings.expand(bs, -1)

        # SCS NFN SS
        SCS = torch.cat((x_t2, x_axial_center, x_stack, level_embeddings), dim=1)

        LNFN = torch.cat((x_t1_l, x_axial_left, x_stack, level_embeddings), dim=1)
        RNFN = torch.cat((x_t1_r, x_axial_right, x_stack, level_embeddings), dim=1)

        SS_left = torch.cat((x_axial_left, x_t1_l, x_stack, level_embeddings), dim=1)
        SS_right = torch.cat((x_axial_right, x_t1_r, x_stack, level_embeddings), dim=1)

        x_scs = self.scs(SCS)

        x_nfn_left = self.nfn_l(LNFN)
        x_nfn_right = self.nfn_r(RNFN)

        x_ss_left = self.ss_left(SS_left)
        x_ss_right = self.ss_right(SS_right)

        x = torch.cat([x_scs, x_nfn_left, x_nfn_right, x_ss_left, x_ss_right], dim=1)
        

        # Final output (15 ordinal classes and combined output)
        assert x.shape[1]==15, f"Expected 15 classes, got {x.shape[1]}"
        return x
    
=======
        self.num_classes = num_classes
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
        assert x.shape[1] == self.num_classes
        return x
>>>>>>> main
=======
        self.num_classes = num_classes
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
        assert x.shape[1] == self.num_classes
        return x
>>>>>>> main
