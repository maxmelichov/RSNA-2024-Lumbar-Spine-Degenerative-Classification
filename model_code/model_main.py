from torchvision import models
import torch
import os
import sys
from abc import ABC
from pathlib import Path
from tqdm import tqdm
import inspect
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score , roc_auc_score
from torch.utils.data import DataLoader
import os
from pathlib import Path
import inspect
from tqdm import tqdm
import numpy as np
import statistics
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
from typing import Any
import pandas as pd
from sklearn.model_selection import KFold
# from apex import amp

# from tensorboardX import SummaryWriter
from torch.nn import DataParallel
import time
import json
import torch.nn.functional as F
from model_code.utils import WeightedLosses, AverageMeter, FocalLoss, FocalLossWithWeights, SevereLoss
from model_code.utilsViT.utils import Attention_Correlation_weight_reshape_loss, fit_inv_covariance, mahalanobis_distance
from model_code.data_loader import data_loader
from model_code.custom_model import CustomRain
from sklearn.model_selection import StratifiedKFold


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Abstract(ABC):
    """
    Abstract class for models.

    Args:
        model (nn.modules): The model to be trained.
        epochs (int, optional): The number of epochs for training. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 128.
        save_wieghts (bool, optional): Whether to save the weights of the model. Defaults to False.
        load_weights (str, optional): The path to the pre-trained model weights. Defaults to None.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
        scheduler_lr (float, optional): The learning rate that the scheduler will reduce to. Defaults to 1e-6.
    """

    def __init__(self, model_l1_l2: nn.modules, model_l2_l3: nn.modules, model_l3_l4: nn.modules, model_l4_l5: nn.modules,
                model_l5_s1: nn.modules, epochs:int = 10, batch_size:int = 128, save_wieghts:bool = False,
                load_weights:str = None, opt: Any = None, lr: float = 1e-4, scheduler_lr: float = 1e-6):
        super().__init__()
        self.model_l1_l2 = model_l1_l2
        self.model_l2_l3 = model_l2_l3
        self.model_l3_l4 = model_l3_l4
        self.model_l4_l5 = model_l4_l5
        self.model_l5_s1 = model_l5_s1
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_wieghts = save_wieghts
        self.load_weights = load_weights
        self.Validation_Accuracy = 0.0
        self.name = Abstract.who_called_me()
        self.lr = lr
        # this is the learning rate that the scheduler will reduce to
        self.scheduler_lr = scheduler_lr
        self.opt = opt

    def plot_metrics(self, losses: list, accs: list, val_losses: list, val_acc: list) -> None:
        """
        Plot the training and validation losses, and training and validation accuracies.

        Args:
            losses (list): List of training losses.
            accs (list): List of training accuracies.
            val_losses (list): List of validation losses.
            val_acc (list): List of validation accuracies.
        """
        self.epochs = range(1, len(losses) + 1)

        plt.figure(figsize=(12, 4))

        # Plot training and validation losses
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, losses, 'b', label='Training loss')
        plt.plot(self.epochs, val_losses, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, accs, 'b', label='Training accuracy')
        plt.plot(self.epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


    @staticmethod
    def who_called_me() -> str:
        """
        Get the name of the class that called this method.

        Returns:
            str: The name of the class.
        """
        stack = inspect.stack()
        return stack[1][0].f_locals["self"].__class__.__name__


    @staticmethod
    def count_parameters(model:nn.Module) -> int:
        """
        Count the number of trainable parameters in the model.

        Args:
            model (nn.Module): The model.

        Returns:
            int: The number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    @staticmethod
    def save_model(model: nn.Module, path: Path):
        """
        Save the model's state dictionary to a file.

        Args:
            model (nn.Module): The model.
            path (Path): The path to save the model.
        """
        torch.save(model.state_dict(), path)


    @staticmethod
    def load_model(model: nn.Module, path: Path) -> nn.Module:
        """
        Load the model's state dictionary from a file.

        Args:
            model (nn.Module): The model.
            path (Path): The path to load the model from.

        Returns:
            nn.Module: The loaded model.
        """
        model.load_state_dict(torch.load(path))
        return model


    @staticmethod
    def get_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model.

        Args:
            model (nn.Module): The model.
            lr (float): The learning rate for the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.AdamW(model.parameters(), lr=lr)
    
    @staticmethod
    def get_scheduler_reduceOnPlateau(optimizer: torch.optim.Optimizer,
                                      mode: str, factor: float, patience: int,
                                      threshold: float) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """
        Get the learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            mode (str): The mode for the scheduler.
            factor (float): The factor by which the learning rate will be reduced.
            patience (int): The number of epochs with no improvement after which the learning rate will be reduced.
            threshold (float): The threshold for measuring the new optimum.

        Returns:
            torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, verbose=True)


    # def validate(self, loss_fn: nn.Module, val_loader: DataLoader) -> tuple[float, float]:
    #     """
    #     Validate the model on the validation set.

    #     Args:
    #         loss_fn (nn.Module): The loss function.
    #         val_loader (DataLoader): The validation data loader.

    #     Returns:
    #         tuple[float, float]: The mean loss and validation accuracy.
    #     """
    #     self.model.eval()
    #     correct = 0.0
    #     number_steps = 0
    #     iter_per_epoch = len(val_loader)
    #     max_iter = self.opt.niter * iter_per_epoch
    #     with torch.no_grad():
    #         losses = AverageMeter() 
    #         validation_batch_losses = []
    #         progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    #         for iter_num, (stack, labels) in enumerate(progress_bar):
    #             # Axial_T2 = Axial_T2.cuda()
    #             # Sagittal_T1 = Sagittal_T1.cuda()
    #             # Sagittal_T2_STIR = Sagittal_T2_STIR.cuda()
    #             # category_hot = category_hot.cuda()
    #             stack = stack.cuda()
    #             img_label = np.array(labels)
    #             labels = labels.cuda().to(torch.long)
    #             step = iter_num / max_iter
    #             outputs, _, _ = self.model(stack, step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk,
    #                                          k=self.opt.k_weight, thr=self.opt.k_thr)
    #             outputs = outputs.reshape(-1, 3, 25)
    #             loss_dis = loss_fn(outputs, labels)
    #             loss_total = loss_dis
    #             losses.update(loss_total.item())
    #             progress_bar.set_postfix({'loss':losses.avg})
    #     print("Validation Loss: ", losses.avg)
    #     return losses.avg

    # def train(self) -> None:
    #     """
    #     Train the model.
    #     """
    #     if self.load_weights != None:
    #         self.model = Abstract.load_model(self.model, self.load_weights)
    #     [c_cross, c_in, c_out] = [nn.Parameter(torch.tensor(float(i)).cuda()) for i in self.opt.c_initilize.split()]
    #     min_threshold_H = [int(i) for i in self.opt.min_threshold_H.split()]
    #     min_threshold = [int(i) for i in self.opt.min_threshold.split()]
    #     attention_loss = Attention_Correlation_weight_reshape_loss(c_out=c_out, c_in=c_in, c_cross=c_cross)
        

    #     optimizer_dict = [{"params": self.model.parameters(), 'lr': self.opt.lr},
    #                     {"params": [c_in, c_out, c_cross], 'lr': self.opt.lr}]
    #     optimizer = torch.optim.AdamW(optimizer_dict, lr=self.opt.lr, betas=(self.opt.beta1, 0.999), eps=self.opt.eps)
    #     print("Number of trainable parameters:", Abstract.count_parameters(self.model))
    #     scheduler = Abstract.get_scheduler_reduceOnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001)
    #     weights = torch.tensor([1.0, 2.0, 4.0])
    #     # criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    #     criterion = SevereLoss()
    #     # criterion = FocalLossWithWeights()
    #     criterion = criterion.cuda()
    #     loss_fn, weights = [], []
    #     for loss_name, weight in {criterion:1}.items():
    #         loss_fn.append(loss_name.cuda())
    #         weights.append(weight)
    #     weightedloss = WeightedLosses(loss_fn, weights)
    #     # summary_writer = SummaryWriter(self.name + "_logs" + str(time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())))
    #     loss_functions = {"classifier_loss": weightedloss}
    #     train_losses, train_accs = [],[]
    #     best_valid_loss = 0.0
    #     early_stopping_counter = 0
        
    #     feat_tensorlist_normal = []
    #     feat_tensorlist_mild = []
    #     feat_tensorlist_severe = []


    #     train_descriptions = r"csv\train_series_descriptions_with_paths.csv"
    #     train_path = r"train.csv"
    #     train_dataset = data_loader(train_path, train_descriptions)
    #     train_df = pd.read_csv(train_path)
    #     primary_labels = train_df['study_id'].values
    #     skf = KFold(n_splits=5, shuffle=True, random_state=42)
    #     autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
    #     scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=4096)


    #     for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(primary_labels)), primary_labels)):
    #         print(f"Fold: {fold + 1}")
    #         train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    #         val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    #         train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle = True)
    #         val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle = True)
            
    #         iter_per_epoch = len(train_loader)
    #         max_iter = self.opt.niter * iter_per_epoch

    #         for current_epoch in range(self.epochs):
    #             print('Epoch {}/{}'.format(current_epoch+1, self.epochs))
    #             running_loss = 0.0
    #             correct = 0.0
    #             progress_bar = tqdm(train_loader, desc='Training', leave=False)
    #             number_steps = 0
    #             losses = AverageMeter() 
    #             for iter_num, (stack, labels) in enumerate(progress_bar):
    #                 if (iter_num != 0 and iter_num % int(self.opt.update_epoch * iter_per_epoch) == 0):
    #                     ## for compute MVG parameter
    #                     feat_tensorlist_normal = torch.cat(feat_tensorlist_normal, dim=0)
    #                     inv_covariance_normal = fit_inv_covariance(feat_tensorlist_normal).cuda() # maybe try detach
    #                     mean_normal = feat_tensorlist_normal.mean(dim=0).cpu()  # mean features.
    #                     gauss_param_normal = {'mean': mean_normal.tolist(), 'covariance': inv_covariance_normal.tolist()}
    #                     with open(os.path.join(self.opt.outf, 'gauss_param_real.json'),'w') as f:
    #                         json.dump(gauss_param_normal, f)
    #                     feat_tensorlist_normal = []

    #                     feat_tensorlist_mild = torch.cat(feat_tensorlist_mild, dim=0)
    #                     inv_covariance_mild = fit_inv_covariance(feat_tensorlist_mild).cuda() # maybe try detach
    #                     mean_mild = feat_tensorlist_mild.mean(dim=0).cpu()  # mean features.
    #                     gauss_param_mild = {'mean': mean_mild.tolist(), 'covariance': inv_covariance_mild.tolist()}
    #                     with open(os.path.join(self.opt.outf, 'gauss_param_fake.json'),'w') as f:
    #                         json.dump(gauss_param_mild, f)
    #                     feat_tensorlist_mild = []

    #                     feat_tensorlist_severe = torch.cat(feat_tensorlist_severe, dim=0)
    #                     inv_covariance_severe = fit_inv_covariance(feat_tensorlist_severe).cuda() # maybe try detach
    #                     mean_severe = feat_tensorlist_severe.mean(dim=0).cpu()  # mean features.
    #                     gauss_param_severe = {'mean': mean_severe.tolist(), 'covariance': inv_covariance_severe.tolist()}
    #                     with open(os.path.join(self.opt.outf, 'gauss_param_fake.json'),'w') as f:
    #                         json.dump(gauss_param_severe, f)
    #                     feat_tensorlist_severe = []


                    
    #                 self.model.train(True)
    #                 # Axial_T2 = Axial_T2.cuda()
    #                 # Sagittal_T1 = Sagittal_T1.cuda()
    #                 # Sagittal_T2_STIR = Sagittal_T2_STIR.cuda()
    #                 # category_hot = category_hot.cuda()
                    
    #                 img_label = labels.numpy().astype(np.float64)
    #                 stack = stack.cuda()
    #                 labels = labels.cuda().to(torch.long)
    #                 step = iter_num / max_iter
    #                 optimizer.zero_grad()
    #                 with autocast:
    #                     classes, feat_patch, attn_map = self.model(stack,
    #                                                                 step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk,
    #                                                                 k=self.opt.k_weight, thr=self.opt.k_thr)
                        
    #                     classes = classes.reshape(-1, 3, 25)

    #                     ### learn MVG parameters
    #                     normalindex = np.where(img_label==0.0)[0]
    #                     attn_map_normal = torch.sigmoid(torch.mean(attn_map[normalindex,:, 1:, 1:], dim=1))
    #                     feat_patch_normal = feat_patch[normalindex[:4]]
    #                     B, H, W, C = feat_patch_normal.size()
    #                     feat_tensorlist_normal.append(feat_patch_normal.reshape(B*H*W, C).cpu().detach())

    #                     mildindex = np.where(img_label==1.0)[0]
    #                     feat_patch_mild = feat_patch[mildindex[:4]]
    #                     attn_map_mild = torch.sigmoid(torch.mean(attn_map[mildindex,:, 1:, 1:], dim=1))
    #                     # attn_map_fake = torch.mean(attn_map[fakeindex,:, 1:, 1:], dim=1)

    #                     severeindex = np.where(img_label==1.0)[0]
    #                     feat_patch_severe = feat_patch[severeindex[:4]]
    #                     attn_map_severe = torch.sigmoid(torch.mean(attn_map[severeindex,:, 1:, 1:], dim=1))

    #                     del attn_map 

    #                     feat_patch_mild_inner = feat_patch_mild[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :]
    #                     B, H, W, C = feat_patch_mild_inner.size()
    #                     feat_tensorlist_mild.append(feat_patch_mild_inner.reshape(B*H*W, C).cpu().detach())

    #                     feat_patch_severe_inner = feat_patch_severe[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :]
    #                     B, H, W, C = feat_patch_severe_inner.size()
    #                     feat_tensorlist_severe.append(feat_patch_severe_inner.reshape(B*H*W, C).cpu().detach())  
    
    #                     if current_epoch > 0 and iter_num >= int(self.opt.update_epoch * iter_per_epoch) and self.opt.lambda_corr > 0:
    #                         B, H, W, C = feat_patch.size()
    #                         maha_patch_normal = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_normal.cuda(), inv_covariance_normal.cuda())
    #                         maha_patch_mild = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_mild.cuda(), inv_covariance_mild.cuda())
    #                         maha_patch_severe = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_severe.cuda(), inv_covariance_severe.cuda())
    #                         del feat_patch
                            
    #                         index_map_mild = torch.relu(maha_patch_normal - maha_patch_mild).reshape((B, H, W))
    #                         index_map_severe = torch.relu(maha_patch_normal - maha_patch_severe).reshape((B, H, W))

    #                         loss_dis = criterion(classes, labels)
    #                         ### for attetion correlation loss
    #                         loss_inter_frame = attention_loss(attn_map_normal, attn_map_mild, attn_map_severe, index_map_mild[mildindex,:], index_map_severe[severeindex,:])
    #                         lambda_c = [float(p) for p in self.opt.lambda_c_param.split()]
    #                         loss_dis_total = loss_dis + self.opt.lambda_corr * loss_inter_frame + lambda_c[0]*torch.abs(c_cross) + lambda_c[1]/torch.abs(c_in) + lambda_c[2]/torch.abs(c_out)

    #                     else:
    #                         loss_dis = criterion(classes, labels)
    #                         loss_dis_total = loss_dis

    #                 scaler.scale(loss_dis_total).backward()
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #                 running_loss += loss_dis_total.item()
    #                 number_steps += 1
    #                 losses.update(loss_dis_total.item())
    #                 progress_bar.set_postfix({'epoch': current_epoch, 'loss': losses.avg})
    #             torch.cuda.synchronize()
    #             train_losses.append(running_loss / (len(train_loader) * self.batch_size))
    #             train_accs.append(100 * (correct / number_steps))
    #             valid_loss = self.validate(criterion, val_loader)
    #             scheduler.step(valid_loss)
    #             print("Last LR", scheduler.get_last_lr())
    #             print('Train Loss: {:.4f}'.format(losses.avg))
    #             print('--------------------------------')
    #             # summary_writer.add_scalar('train loss', float(losses.avg), global_step=current_epoch)
    #             # summary_writer.add_scalar('train acc', float(100*(correct/number_steps)), global_step=current_epoch)
    #             # summary_writer.add_scalar('validation loss', float(valid_loss), global_step=current_epoch)
    #             # summary_writer.add_scalar('validation acc', float(valid_acc), global_step=current_epoch)
    #             # Check for overfitting
    #             if valid_loss > best_valid_loss:
    #                 best_val_acc = valid_loss
    #                 self.Validation_loss = best_val_acc
                    
    #             else:
    #                 early_stopping_counter += 1

    #             if early_stopping_counter >= 5:  # Stop if validation loss doesn't improve for 2 epochs
    #                 print("Early stopping due to overfitting")
    #                 break
                
    #             if self.save_wieghts and (early_stopping_counter == 0 or valid_loss > best_valid_loss):
    #                 self.save_model(self.model, self.name + "_best_validation.pt")
    #         # self.plot_metrics(losses, accs, val_losses, val_acc)
    #         if self.save_wieghts:
    #             os.rename(self.name + "_best_validation.pt", self.name + "_"+ str(self.Validation_loss)+ "_fold_" + str(fold+1) + ".pt")

    def validate(self, loss_fn: nn.Module, val_loader: DataLoader, autocast: torch.cuda.amp.autocast) -> tuple[float, float]:
        """
        Validate the model on the validation set.

        Args:
            loss_fn (nn.Module): The loss function.
            val_loader (DataLoader): The validation data loader.

        Returns:
            tuple[float, float]: The mean loss and validation accuracy.
        """
        self.model_l1_l2.eval()
        self.model_l2_l3.eval()
        self.model_l3_l4.eval()
        self.model_l4_l5.eval()
        self.model_l5_s1.eval()
        with torch.no_grad():
            losses = AverageMeter() 
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for (sagittal_l1_l2, sagittal_l2_l3, sagittal_l3_l4, sagittal_l4_l5, sagittal_l5_s1, axial_l1_l2, axial_l2_l3, axial_l3_l4, axial_l4_l5, axial_l5_s1, labels) in progress_bar:
                sagittal_l1_l2 = sagittal_l1_l2.cuda()
                sagittal_l2_l3 = sagittal_l2_l3.cuda()
                sagittal_l3_l4 = sagittal_l3_l4.cuda()
                sagittal_l4_l5 = sagittal_l4_l5.cuda()
                sagittal_l5_s1 = sagittal_l5_s1.cuda()
                axial_l1_l2 = axial_l1_l2.cuda()
                axial_l2_l3 = axial_l2_l3.cuda()
                axial_l3_l4 = axial_l3_l4.cuda()
                axial_l4_l5 = axial_l4_l5.cuda()
                axial_l5_s1 = axial_l5_s1.cuda()
                labels = labels.cuda().to(torch.long)
                loss_dis = 0.0
                with autocast:
                    outputs_l1_l2 = self.model_l1_l2(sagittal_l1_l2)
                    outputs_l2_l3 = self.model_l2_l3(sagittal_l2_l3)
                    outputs_l3_l4 = self.model_l3_l4(sagittal_l3_l4)
                    outputs_l4_l5 = self.model_l4_l5(sagittal_l4_l5)
                    outputs_l5_s1 = self.model_l5_s1(sagittal_l5_s1)
                    outputs = outputs_l1_l2 + outputs_l2_l3 + outputs_l3_l4 + outputs_l4_l5 + outputs_l5_s1
                    
                    
                    # outputs = outputs.reshape(-1, 3, 25)
                    for col in range(25):
                        pred = outputs[:, col * 3:col * 3 + 3]
                        gt = labels[:, col]
                        loss_dis = loss_dis + loss_fn(pred, gt) / 25 
                loss_total = loss_dis
                losses.update(loss_total.item())
                progress_bar.set_postfix({'loss':losses.avg})
        print("Validation Loss: ", losses.avg)
        return losses.avg
    
    def train(self) -> None:
        """
        Train the model.
        """
        optimizer_l1_l2 = torch.optim.AdamW(self.model_l1_l2.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer_l2_l3 = torch.optim.AdamW(self.model_l2_l3.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer_l3_l4 = torch.optim.AdamW(self.model_l3_l4.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer_l4_l5 = torch.optim.AdamW(self.model_l4_l5.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer_l5_s1 = torch.optim.AdamW(self.model_l5_s1.parameters(), lr=self.lr, weight_decay=1e-4)

        print("Number of trainable parameters:", Abstract.count_parameters(self.model_l1_l2)*5)
        scheduler_l1_l2 = Abstract.get_scheduler_reduceOnPlateau(optimizer_l1_l2, mode='min', factor=0.1, patience=2, threshold=0.0001)
        scheduler_l2_l3 = Abstract.get_scheduler_reduceOnPlateau(optimizer_l2_l3, mode='min', factor=0.1, patience=2, threshold=0.0001)
        scheduler_l3_l4 = Abstract.get_scheduler_reduceOnPlateau(optimizer_l3_l4, mode='min', factor=0.1, patience=2, threshold=0.0001)
        scheduler_l4_l5 = Abstract.get_scheduler_reduceOnPlateau(optimizer_l4_l5, mode='min', factor=0.1, patience=2, threshold=0.0001)
        scheduler_l5_s1 = Abstract.get_scheduler_reduceOnPlateau(optimizer_l5_s1, mode='min', factor=0.1, patience=2, threshold=0.0001)

        weights = torch.tensor([1.0, 2.0, 4.0], dtype= torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        # criterion = SevereLoss(temperature=0)
        # criterion = CustomLoss(weights.to(device))
        # criterion = FocalLossWithWeights()
        criterion = criterion.cuda()
        loss_fn, weights = [], []
        for loss_name, weight in {criterion:1}.items():
            loss_fn.append(loss_name.cuda())
            weights.append(weight)
        # summary_writer = SummaryWriter(self.name + "_logs" + str(time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())))
        train_losses = []
        
        train_labels = r"csv\train_series_descriptions_with_paths.csv"
        train_path = r"train.csv"
        train_descriptions = r"train_series_descriptions.csv"
        train_dataset = data_loader(train_path, train_labels, train_descriptions)
        train_df = pd.read_csv(train_path)
        primary_labels = train_df['study_id'].values
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=4096)

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(primary_labels)))):
            print(f"Fold: {fold + 1}")
            best_valid_loss = np.inf
            early_stopping_counter = 0
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle = True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

            for current_epoch in range(self.epochs):
                print('Epoch {}/{}'.format(current_epoch+1, self.epochs))
                running_loss = 0.0
                progress_bar = tqdm(train_loader, desc='Training', leave=False)
                losses = AverageMeter() 
                for (sagittal_l1_l2, sagittal_l2_l3, sagittal_l3_l4, sagittal_l4_l5, sagittal_l5_s1, axial_l1_l2, axial_l2_l3, axial_l3_l4, axial_l4_l5, axial_l5_s1, labels) in progress_bar:
                    # print("stack_l1_l2", stack_l1_l2.shape)
                    self.model_l1_l2.train(True)
                    self.model_l2_l3.train(True)
                    self.model_l3_l4.train(True)
                    self.model_l4_l5.train(True)
                    self.model_l5_s1.train(True)
                    sagittal_l1_l2 = sagittal_l1_l2.cuda()
                    sagittal_l2_l3 = sagittal_l2_l3.cuda()
                    sagittal_l3_l4 = sagittal_l3_l4.cuda()
                    sagittal_l4_l5 = sagittal_l4_l5.cuda()
                    sagittal_l5_s1 = sagittal_l5_s1.cuda()
                    axial_l1_l2 = axial_l1_l2.cuda()
                    axial_l2_l3 = axial_l2_l3.cuda()
                    axial_l3_l4 = axial_l3_l4.cuda()
                    axial_l4_l5 = axial_l4_l5.cuda()
                    axial_l5_s1 = axial_l5_s1.cuda()


                    labels = labels.cuda().to(torch.long)
                    optimizer_l1_l2.zero_grad()
                    optimizer_l2_l3.zero_grad()
                    optimizer_l3_l4.zero_grad()
                    optimizer_l4_l5.zero_grad()
                    optimizer_l5_s1.zero_grad()
                    loss_l1_l2 = 0.0
                    loss_l2_l3 = 0.0
                    loss_l3_l4 = 0.0
                    loss_l4_l5 = 0.0
                    loss_l5_s1 = 0.0
                    with autocast:
                        output_l1_l2 = self.model_l1_l2(sagittal_l1_l2, axial_l1_l2)
                        output_l2_l3 = self.model_l2_l3(sagittal_l2_l3, axial_l2_l3)
                        output_l3_l4 = self.model_l3_l4(sagittal_l3_l4, axial_l3_l4)
                        output_l4_l5 = self.model_l4_l5(sagittal_l4_l5, axial_l4_l5)
                        output_l5_s1 = self.model_l5_s1(sagittal_l5_s1, axial_l5_s1)
                        # outputs = outputs.reshape(-1, 3, 25)
                        col_i = 0
                        for col in range(5):
                            pred = output_l1_l2[:,col*3:col*3+3]
                            gt = labels[:,col_i]
                            col_i += 1
                            loss_l1_l2 = loss_l1_l2 + criterion(pred, gt) / 5
                        scaler.scale(loss_l1_l2).backward()
                        scaler.unscale_(optimizer_l1_l2)
                        torch.nn.utils.clip_grad_norm_(self.model_l1_l2.parameters(), 2.0)
                        scaler.step(optimizer_l1_l2)
                        scaler.update()

                        for col in range(5):
                            pred = output_l2_l3[:,col*3:col*3+3]
                            gt = labels[:,col_i]
                            col_i += 1
                            loss_l2_l3 = loss_l2_l3 + criterion(pred, gt) / 5
                        scaler.scale(loss_l2_l3).backward()
                        scaler.unscale_(optimizer_l2_l3)
                        torch.nn.utils.clip_grad_norm_(self.model_l2_l3.parameters(), 2.0)
                        scaler.step(optimizer_l2_l3)
                        scaler.update()

                        for col in range(5):
                            pred = output_l3_l4[:,col*3:col*3+3]
                            gt = labels[:,col_i]
                            col_i += 1
                            loss_l3_l4 = loss_l3_l4 + criterion(pred, gt) / 5
                        scaler.scale(loss_l3_l4).backward()
                        scaler.unscale_(optimizer_l3_l4)
                        torch.nn.utils.clip_grad_norm_(self.model_l3_l4.parameters(), 2.0)
                        scaler.step(optimizer_l3_l4)
                        scaler.update()

                        for col in range(5):
                            pred = output_l4_l5[:,col*3:col*3+3]
                            gt = labels[:,col]
                            col_i += 1
                            loss_l4_l5 = loss_l4_l5 + criterion(pred, gt) / 5
                        scaler.scale(loss_l4_l5).backward()
                        scaler.unscale_(optimizer_l4_l5)
                        torch.nn.utils.clip_grad_norm_(self.model_l4_l5.parameters(), 2.0)
                        scaler.step(optimizer_l4_l5)
                        scaler.update()
                        
                        for col in range(5):
                            pred = output_l5_s1[:,col*3:col*3+3]
                            gt = labels[:,col]
                            col_i += 1
                            loss_l5_s1 = loss_l5_s1 + criterion(pred, gt) / 5
                        scaler.scale(loss_l5_s1).backward()
                        scaler.unscale_(optimizer_l5_s1)
                        torch.nn.utils.clip_grad_norm_(self.model_l5_s1.parameters(), 2.0)
                        scaler.step(optimizer_l5_s1)
                        scaler.update()

                    loss_dis_total = (loss_l1_l2 + loss_l2_l3 + loss_l3_l4 + loss_l4_l5 + loss_l5_s1) / 5
                    
                    running_loss += loss_dis_total.item()
                    losses.update(loss_dis_total.item())
                    progress_bar.set_postfix({'epoch': current_epoch, 'loss': losses.avg})
                train_losses.append(running_loss / (len(train_loader) * self.batch_size))
                valid_loss = self.validate(criterion, val_loader, autocast)
                scheduler_l2_l3.step(valid_loss)
                scheduler_l3_l4.step(valid_loss)
                scheduler_l4_l5.step(valid_loss)
                scheduler_l5_s1.step(valid_loss)
                scheduler_l1_l2.step(valid_loss)

                print("Last LR", scheduler_l2_l3.get_last_lr())
                print('Train Loss: {:.4f}'.format(losses.avg))
                print('--------------------------------')
                # Check for overfitting
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.Validation_loss = best_valid_loss
                    
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= 5:  # Stop if validation loss doesn't improve for 2 epochs
                    print("Early stopping due to overfitting")
                    break
                
                if self.save_wieghts and (early_stopping_counter == 0 or valid_loss > best_valid_loss):
                    self.save_model(self.model_l1_l2, self.name + "_best_validation_l1_l2.pt")
                    self.save_model(self.model_l2_l3, self.name + "_best_validation_l2_l3.pt")
                    self.save_model(self.model_l3_l4, self.name + "_best_validation_l3_l4.pt")
                    self.save_model(self.model_l4_l5, self.name + "_best_validation_l4_l5.pt")
                    self.save_model(self.model_l5_s1, self.name + "_best_validation_l5_s1.pt")
            if self.save_wieghts:
                os.rename(self.name + "_best_validation_l1_l2.pt", self.name + "_" + str(self.Validation_loss)+ "_fold_" + str(fold+1) + ".pt")
                os.rename(self.name + "_best_validation_l2_l3.pt", self.name + "_" + str(self.Validation_loss)+ "_fold_" + str(fold+1) + ".pt")
                os.rename(self.name + "_best_validation_l3_l4.pt", self.name + "_" + str(self.Validation_loss)+ "_fold_" + str(fold+1) + ".pt")
                os.rename(self.name + "_best_validation_l4_l5.pt", self.name + "_" + str(self.Validation_loss)+ "_fold_" + str(fold+1) + ".pt")
                os.rename(self.name + "_best_validation_l5_s1.pt", self.name + "_" + str(self.Validation_loss)+ "_fold_" + str(fold+1) + ".pt")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, type_) -> None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_true, y_pred), display_labels = classes)
        cm_display.plot()
        cm_display.ax_.set_title(type_ + ' Confusion Matrix')

class RainDrop(Abstract):
    def __init__(self, num_classes =75, epochs: int = 10, batch_size:int = 128, save_wieghts:bool = False, load_weights:str = None) -> None:
        self.model_l1_l2 = CustomRain(num_classes, True).to(device)
        self.model_l2_l3 = CustomRain(num_classes, True).to(device)
        self.model_l3_l4 = CustomRain(num_classes, True).to(device)
        self.model_l4_l5 = CustomRain(num_classes, True).to(device)
        self.model_l5_s1 = CustomRain(num_classes, True).to(device)

        super().__init__(self.model_l1_l2, self.model_l2_l3, self.model_l3_l4, self.model_l4_l5, self.model_l5_s1,
                          epochs, batch_size, save_wieghts, load_weights, lr=3.6e-5)

    def __call__(self) -> None:
        self.train()
