from torchvision import models
import torch
import os
import sys
from torch.utils.data import DataLoader
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
# from apex import amp

# from tensorboardX import SummaryWriter
from torch.nn import DataParallel
import time
import json
import torch.nn.functional as F
from model_code.utils import WeightedLosses, AverageMeter, FocalLoss, FocalLossWithWeights
from model_code.utilsViT.utils import Attention_Correlation_weight_reshape_loss, fit_inv_covariance, mahalanobis_distance
from model_code.data_loader import data_loader
from model_code.custom_model import CustomModel
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

    def __init__(self, model: nn.modules, epochs:int = 10, batch_size:int = 128, save_wieghts:bool = False, load_weights:str = None, opt: Any = None, lr: float = 1e-4, scheduler_lr: float = 1e-6):
        super().__init__()
        self.model = model
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


    def validate(self, loss_fn: nn.Module, val_loader: DataLoader) -> tuple[float, float]:
        """
        Validate the model on the validation set.

        Args:
            loss_fn (nn.Module): The loss function.
            val_loader (DataLoader): The validation data loader.

        Returns:
            tuple[float, float]: The mean loss and validation accuracy.
        """
        self.model.eval()
        correct = 0.0
        number_steps = 0
        iter_per_epoch = len(val_loader)
        max_iter = self.opt.niter * iter_per_epoch
        with torch.no_grad():
            loss_dis = 0.0
            validation_batch_losses = []
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for iter_num, (Axial_T2, Sagittal_T1, Sagittal_T2_STIR, category_hot, labels) in enumerate(progress_bar):
                Axial_T2 = Axial_T2.cuda()
                Sagittal_T1 = Sagittal_T1.cuda()
                Sagittal_T2_STIR = Sagittal_T2_STIR.cuda()
                category_hot = category_hot.cuda()
                img_label = np.array(labels)
                labels = labels.cuda().to(torch.float)
                step = iter_num / max_iter
                outputs, _, _ = self.model(Axial_T2 = Axial_T2, Sagittal_T1 = Sagittal_T1, Sagittal_T2_STIR = Sagittal_T2_STIR,
                                             category_hot = category_hot, step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk,
                                             k=self.opt.k_weight, thr=self.opt.k_thr)

                for col in range(5):
                    pred = outputs[:,col*3:col*3+3]
                    
                    gt = labels[:,col*3:col*3+3]
                    # Create a mask where all elements in the row are True if they match the target
                    mask = (gt == torch.tensor([-100, -100, -100], device='cuda:0')).all(dim=1)

                    # Use the mask to filter rows that do NOT match the target in both gt and pred
                    gt = gt[~mask]
                    pred = pred[~mask]
                    loss_dis = loss_dis + loss_fn(pred, gt) / 5
                loss_total = loss_dis
                validation_batch_losses.append(float(loss_total))
                mean_loss = statistics.mean(validation_batch_losses)
                progress_bar.set_postfix({'loss':mean_loss})
        print("Validation Loss: ", mean_loss)
        return mean_loss

    def train(self) -> None:
        """
        Train the model.
        """
        if self.load_weights != None:
            self.model = Abstract.load_model(self.model, self.load_weights)
        [c_cross, c_in, c_out] = [nn.Parameter(torch.tensor(float(i)).cuda()) for i in self.opt.c_initilize.split()]
        min_threshold_H = [int(i) for i in self.opt.min_threshold_H.split()]
        min_threshold = [int(i) for i in self.opt.min_threshold.split()]
        attention_loss = Attention_Correlation_weight_reshape_loss(c_out=c_out, c_in=c_in, c_cross=c_cross)

        

        optimizer_dict = [{"params": self.model.parameters(), 'lr': self.opt.lr},
                        {"params": [c_in, c_out, c_cross], 'lr': self.opt.lr}]
        optimizer = torch.optim.AdamW(optimizer_dict, lr=self.opt.lr, betas=(self.opt.beta1, 0.999), eps=self.opt.eps)
        print("Number of trainable parameters:", Abstract.count_parameters(self.model))
        scheduler = Abstract.get_scheduler_reduceOnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001)
        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        # criterion = FocalLossWithWeights()
        criterion = criterion.cuda()
        loss_fn, weights = [], []
        for loss_name, weight in {criterion:1}.items():
            loss_fn.append(loss_name.cuda())
            weights.append(weight)
        weightedloss = WeightedLosses(loss_fn, weights)
        # summary_writer = SummaryWriter(self.name + "_logs" + str(time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())))
        loss_functions = {"classifier_loss": weightedloss}
        train_losses, train_accs = [],[]
        best_valid_loss = 0.0
        early_stopping_counter = 0
        
        feat_tensorlist = []
        feat_tensorlist_fake = []


        labels_path = r"csv\labels.csv"
        train_data = r"csv\train_data_with_general_paths.csv"
        train_dataset = data_loader(train_data, labels_path)
        train_df = pd.read_csv(labels_path)
        primary_labels = train_df['study_id'].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(primary_labels)), primary_labels)):
            print(f"Fold: {fold + 1}")
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle = True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle = True)
            
            iter_per_epoch = len(train_loader)
            max_iter = self.opt.niter * iter_per_epoch

            for current_epoch in range(self.epochs):
                print('Epoch {}/{}'.format(current_epoch+1, self.epochs))
                running_loss = 0.0
                correct = 0.0
                progress_bar = tqdm(train_loader, desc='Training', leave=False)
                number_steps = 0
                losses = AverageMeter()
                y_list, y_pred_list = [], []
                for iter_num, (Axial_T2, Sagittal_T1, Sagittal_T2_STIR, category_hot, labels) in enumerate(progress_bar):
                    if (iter_num != 0 and iter_num % int(self.opt.update_epoch * iter_per_epoch) == 0):
                        ## for compute MVG parameter
                        feat_tensorlist = torch.cat(feat_tensorlist, dim=0)
                        inv_covariance_real = fit_inv_covariance(feat_tensorlist).cpu() # maybe try detach
                        mean_real = feat_tensorlist.mean(dim=0).cpu()  # mean features.
                        gauss_param_real = {'mean': mean_real.tolist(), 'covariance': inv_covariance_real.tolist()}
                        with open(os.path.join(self.opt.outf, 'gauss_param_real.json'),'w') as f:
                            json.dump(gauss_param_real, f)
                        feat_tensorlist = []

                        feat_tensorlist_fake = torch.cat(feat_tensorlist_fake, dim=0)
                        inv_covariance_fake = fit_inv_covariance(feat_tensorlist_fake).cpu()
                        mean_fake = feat_tensorlist_fake.mean(dim=0).cpu()  # mean features.
                        gauss_param_fake = {'mean': mean_fake.tolist(), 'covariance': inv_covariance_fake.tolist()}
                        with open(os.path.join(self.opt.outf, 'gauss_param_fake.json'),'w') as f:
                            json.dump(gauss_param_fake, f)
                        feat_tensorlist_fake = []
                        torch.cuda.synchronize()

                    
                    self.model.train(True)
                    Axial_T2 = Axial_T2.cuda()
                    Sagittal_T1 = Sagittal_T1.cuda()
                    Sagittal_T2_STIR = Sagittal_T2_STIR.cuda()
                    category_hot = category_hot.cuda()
                    img_label = np.array(labels).astype(np.float32)
                    labels = labels.cuda().to(torch.float32)
                    step = iter_num / max_iter
                    optimizer.zero_grad()
                    loss_dis = 0.0
                    classes, feat_patch, attn_map = self.model(Axial_T2 = Axial_T2, Sagittal_T1 = Sagittal_T1, Sagittal_T2_STIR = Sagittal_T2_STIR,
                                                               category_hot = category_hot, 
                                                                step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk,
                                                                  k=self.opt.k_weight, thr=self.opt.k_thr)
                    ### learn MVG parameters
                    realindex = np.where(img_label==0.0)[0]
                    attn_map_real = torch.sigmoid(torch.mean(attn_map[realindex,:, 1:, 1:], dim=1))
                    feat_patch_real = feat_patch[realindex[:4]]
                    B, H, W, C = feat_patch_real.size()
                    feat_tensorlist.append(feat_patch_real.reshape(B*H*W, C).cpu().detach())

                    fakeindex = np.where(img_label==1.0)[0]
                    feat_patch_fake = feat_patch[fakeindex[:4]]
                    attn_map_fake = torch.sigmoid(torch.mean(attn_map[fakeindex,:, 1:, 1:], dim=1))
                    # attn_map_fake = torch.mean(attn_map[fakeindex,:, 1:, 1:], dim=1)

                    del attn_map           

                    feat_patch_fake_inner = feat_patch_fake[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :]
                    B, H, W, C = feat_patch_fake_inner.size()
                    feat_tensorlist_fake.append(feat_patch_fake_inner.reshape(B*H*W, C).cpu().detach())

                    if current_epoch > 0 and iter_num >= int(self.opt.update_epoch * iter_per_epoch) and self.opt.lambda_corr > 0:
                        B, H, W, C = feat_patch.size()
                        maha_patch_1 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda())
                        maha_patch_2 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda())
                        del feat_patch
                        index_map = torch.relu(maha_patch_1 - maha_patch_2).reshape((B, H, W))
                        for col in range(5):

                            pred = classes[:,col*3:col*3+3]
                            
                            gt = labels[:,col*3:col*3+3]
                            # Create a mask where all elements in the row are True if they match the target
                            mask = (gt == torch.tensor([-100, -100, -100], device='cuda:0')).all(dim=1)

                            # Use the mask to filter rows that do NOT match the target in both gt and pred
                            gt = gt[~mask]
                            pred = pred[~mask]
                            loss_dis = loss_dis + criterion(pred, gt) / 5

                        ### for attetion correlation loss
                        loss_inter_frame = attention_loss(attn_map_real, attn_map_fake, index_map[fakeindex,:])
                        lambda_c = [float(p) for p in self.opt.lambda_c_param.split()]
                        loss_dis_tatol = loss_dis + self.opt.lambda_corr * loss_inter_frame + lambda_c[0]*torch.abs(c_cross) + lambda_c[1]/torch.abs(c_in) + lambda_c[2]/torch.abs(c_out)
                    else:
                        for col in range(5):
                            pred = classes[:,col*3:col*3+3]
                            
                            gt = labels[:,col*3:col*3+3]
                            # Create a mask where all elements in the row are True if they match the target
                            mask = (gt == torch.tensor([-100, -100, -100], device='cuda:0')).all(dim=1)

                            # Use the mask to filter rows that do NOT match the target in both gt and pred
                            gt = gt[~mask]
                            pred = pred[~mask]
                            loss_dis = loss_dis + criterion(pred, gt) / 5
                        loss_dis_tatol = loss_dis
                    
                    loss_dis_tatol.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss_dis_tatol.item()
                    number_steps += 1
                    losses.update(loss_dis_tatol.item())
                    progress_bar.set_postfix({'epoch': current_epoch, 'loss': losses.avg})
                torch.cuda.synchronize()
                train_losses.append(running_loss / (len(train_loader) * self.batch_size))
                train_accs.append(100 * (correct / number_steps))
                valid_loss = self.validate(loss_functions["classifier_loss"], val_loader)
                scheduler.step(valid_loss)
                print("Last LR", scheduler.get_last_lr())
                print('Train Loss: {:.4f}'.format(losses.avg))
                print('--------------------------------')
                # summary_writer.add_scalar('train loss', float(losses.avg), global_step=current_epoch)
                # summary_writer.add_scalar('train acc', float(100*(correct/number_steps)), global_step=current_epoch)
                # summary_writer.add_scalar('validation loss', float(valid_loss), global_step=current_epoch)
                # summary_writer.add_scalar('validation acc', float(valid_acc), global_step=current_epoch)
                # Check for overfitting
                if valid_loss > best_valid_loss:
                    best_val_acc = valid_loss
                    self.Validation_loss = best_val_acc
                    
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= 5:  # Stop if validation loss doesn't improve for 2 epochs
                    print("Early stopping due to overfitting")
                    break
                
                if self.save_wieghts and (early_stopping_counter == 0 or valid_loss > best_valid_loss):
                    self.save_model(self.model, self.name + "_best_validation.pt")
            # self.plot_metrics(losses, accs, val_losses, val_acc)
            if self.save_wieghts:
                os.rename(self.name + "_best_validation.pt", self.name + "_"+ str(self.Validation_loss) + ".pt")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, type_) -> None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_true, y_pred), display_labels = classes)
        cm_display.plot()
        cm_display.ax_.set_title(type_ + ' Confusion Matrix')
    
class UAIViT(Abstract):
    def __init__(self, opt, num_classes:int = 2, epochs: int = 10, batch_size:int = 128, save_wieghts:bool = False, load_weights:str = None) -> None:
        self.model = CustomModel(num_classes).to(device)

        super().__init__(self.model, epochs, batch_size, save_wieghts, load_weights, opt)

    def __call__(self) -> None:
        self.train()