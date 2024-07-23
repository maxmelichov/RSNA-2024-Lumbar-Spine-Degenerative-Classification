from torchvision import models
import torch
import os
import sys
from torch.utils.data import DataLoader
from abc import ABC
from pathlib import Path
from tqdm import tqdm
import inspect
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader
import os
from pathlib import Path
import inspect
from tqdm import tqdm
import numpy as np
import statistics
import torch.nn as nn
import model_code.custom_model as custom_model
import matplotlib.pyplot as plt
import torch.optim
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
from typing import Any
# from apex import amp

from tensorboardX import SummaryWriter
from torch.nn import DataParallel
import time
from network.vit_16_base_feat_middle_gal_v2 import vit_base_patch16_224
import json
import torch.nn.functional as F

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


    def get_csvs_paths(self) -> tuple[Path, Path]:
        """
        Get the paths to the CSV files for training and validation.

        Returns:
            tuple[Path, Path]: The paths to the training and validation CSV files.
        """
        project_path = os.path.dirname(os.path.dirname(os.getcwd()))
        dir_csv_train = os.path.join(r"csvs/train_data.csv")
        dir_csv_val = os.path.join(r"csvs/val_data.csv")
        return dir_csv_train, dir_csv_val


    
    def get_test_csvs_path(self) -> Path:
        """
        Get the path to the CSV file for testing.

        Returns:
            Path: The path to the test CSV file.
        """
        project_path = os.path.dirname(os.path.dirname(os.getcwd()))
        dir_csv_test = os.path.join(r"csvs/test_data.csv")
        return dir_csv_test


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


    @staticmethod
    def confident_strategy(pred, t:float = 0.5, batch_size: int = 32):
        pred = np.array(pred)
        sz = len(pred)
        fakes = np.count_nonzero(pred > t)
        # 11 frames are detected as fakes with high probability
        if fakes > sz // 2.5 and fakes > np.round(batch_size*0.33):
            return np.mean(pred[pred > t])
        elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
            return np.mean(pred[pred < 0.2])
        else:
            return np.mean(pred)
    
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
            validation_batch_losses = []
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            y_list, y_pred_list = [], []
            for iter_num, (raw, labels) in enumerate(progress_bar):
                raw = raw.cuda()
                img_label = np.array(labels)
                labels = labels.cuda().to(torch.float)
                step = iter_num / max_iter
                outputs, _, _ = self.model(raw, step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk, k=self.opt.k_weight, thr=self.opt.k_thr)
                output_pred = torch.argmax(F.softmax(outputs, dim=1).cpu().data, dim=1)
                predicted = np.round(output_pred)
                y = torch.argmax(torch.tensor(img_label), dim=1)
                y_list.append(y)
                y_pred_list.append(predicted)
                correct += torch.sum(predicted.eq(y)).item()
                loss = loss_fn(outputs, labels)
                validation_batch_losses.append(float(loss))
                mean_loss = statistics.mean(validation_batch_losses)
                number_steps += 32
                progress_bar.set_postfix({'acc':100 * (correct / number_steps)})
        val_acc = 100 * (correct / number_steps)
        print("Validation Accuracy: ", 100 * accuracy_score(y_list, y_pred_list))
        print("Validation Loss: ", mean_loss)
        print("Validation Precision: ", 100 * precision_score(y_list, y_pred_list, labels=[0,1]))
        print("Validation Recall: ", 100 * recall_score(y_list, y_pred_list, labels=[0,1]))
        print("Validation F1 Score: ", 100 * f1_score(y_list, y_pred_list, labels=[0,1]))
        Abstract.plot_confusion_matrix(y_list, y_pred_list, ['Real', 'Fake'], "validation")
        return mean_loss, val_acc

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
        dir_csv_train, dir_csv_val = self.get_csvs_paths()
        optimizer_dict = [{"params": self.model.parameters(), 'lr': self.opt.lr},
                        {"params": [c_in, c_out, c_cross], 'lr': self.opt.lr}]
        optimizer = torch.optim.Adam(optimizer_dict, lr=self.opt.lr, betas=(self.opt.beta1, 0.999), eps=self.opt.eps)
        print("Number of trainable parameters:", Abstract.count_parameters(self.model))
        scheduler = Abstract.get_scheduler_reduceOnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([6, 6/5]).to(device))
        criterion = criterion.cuda()
        loss_fn, weights = [], []
        for loss_name, weight in {criterion:1}.items():
            loss_fn.append(loss_name.cuda())
            weights.append(weight)
        weightedloss = WeightedLosses(loss_fn, weights)
        summary_writer = SummaryWriter(self.name + "_logs" + str(time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())))
        loss_functions = {"classifier_loss": weightedloss}
        train_loader, val_loader = model_loading.data_loader(dir_csv_train, dir_csv_val, self.batch_size)
        train_losses, train_accs = [],[]
        best_val_acc = 0.0
        early_stopping_counter = 0
        iter_per_epoch = len(train_loader)
        max_iter = self.opt.niter * iter_per_epoch
        feat_tensorlist = []
        feat_tensorlist_fake = []
        for current_epoch in range(self.epochs):
            print('Epoch {}/{}'.format(current_epoch+1, self.epochs))
            running_loss = 0.0
            correct = 0.0
            progress_bar = tqdm(train_loader, desc='Training', leave=False)
            number_steps = 0
            losses = AverageMeter()
            y_list, y_pred_list = [], []
            for iter_num, (raw, labels) in enumerate(progress_bar):
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
                image = raw.cuda()
                img_label = np.array(labels).astype(np.float32)
                labels = labels.cuda().to(torch.float32)
                step = iter_num / max_iter
                optimizer.zero_grad()
                classes, feat_patch, attn_map = self.model(image, step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk, k=self.opt.k_weight, thr=self.opt.k_thr)

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
                    
                    loss_dis = criterion(classes, labels)

                    ### for attetion correlation loss
                    loss_inter_frame = attention_loss(attn_map_real, attn_map_fake, index_map[fakeindex,:])
                    lambda_c = [float(p) for p in self.opt.lambda_c_param.split()]
                    loss_dis_tatol = loss_dis + self.opt.lambda_corr * loss_inter_frame + lambda_c[0]*torch.abs(c_cross) + lambda_c[1]/torch.abs(c_in) + lambda_c[2]/torch.abs(c_out)
                else:
                    loss_dis = criterion(classes, labels)
                    loss_dis_tatol = loss_dis
                
                loss_dis_tatol.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                output_pred = torch.argmax(F.softmax(classes, dim=1).cpu().data, dim=1)
                predicted = np.round(output_pred)
                
                y = torch.argmax(torch.tensor(img_label), dim=1)
                y_list.append(y)
                y_pred_list.append(predicted)
                correct += torch.sum(predicted.eq(y)).item()
                running_loss += loss_dis_tatol.item()
                number_steps += 32
                losses.update(loss_dis_tatol.item())
                progress_bar.set_postfix({'epoch': current_epoch, 'loss': losses.avg, 'acc': 100 * (correct / number_steps)})
            torch.cuda.synchronize()
            train_losses.append(running_loss / (len(train_loader) * self.batch_size))
            train_accs.append(100 * (correct / number_steps))
            valid_loss, valid_acc = self.validate(loss_functions["classifier_loss"], val_loader)
            scheduler.step(valid_loss)
            print("Last LR", scheduler.get_last_lr())
            print('Train Accuracy: {:.2f}%'.format(100 * (correct / number_steps)))
            print('Train Loss: {:.4f}'.format(losses.avg))
            print("Precision: ", precision_score(y_list, y_pred_list, labels=[0,1]))
            print("Recall: ", recall_score(y_list, y_pred_list, labels=[0,1]))
            print("F1 Score: ", f1_score(y_list, y_pred_list, labels=[0,1]))
            print('--------------------------------')
            Abstract.plot_confusion_matrix(y_list, y_pred_list, ['Real', 'Fake'], "train")
            summary_writer.add_scalar('train loss', float(losses.avg), global_step=current_epoch)
            summary_writer.add_scalar('train acc', float(100*(correct/number_steps)), global_step=current_epoch)
            summary_writer.add_scalar('validation loss', float(valid_loss), global_step=current_epoch)
            summary_writer.add_scalar('validation acc', float(valid_acc), global_step=current_epoch)
            # Check for overfitting
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                self.Validation_Accuracy = valid_acc
                
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= 5:  # Stop if validation loss doesn't improve for 2 epochs
                print("Early stopping due to overfitting")
                break
            
            if self.save_wieghts and (early_stopping_counter == 0 or valid_acc > best_val_acc):
                self.save_model(self.model, self.name + "_best_validation.pt")
        # self.plot_metrics(losses, accs, val_losses, val_acc)
        if self.save_wieghts:
            os.rename(self.name + "_best_validation.pt", self.name + "_"+ str(self.Validation_Accuracy) + ".pt")
    
    def test(self) -> None:
        """
        Test the model.
        """
        pass
        precision= 0.0
        recall = 0.0
        score = 0.0
        test_acc = 0.0
        number_steps = 0
        self.model = Abstract.load_model(self.model, self.name + "_" + str(self.Validation_Accuracy) + '.pt')
        dir_csv_test = self.get_test_csvs_path()
        test_loader = model_loading.testloader(dir_csv_test,self.batch_size)
        y_true_list, y_pred_list = [], []
        iter_per_epoch = len(test_loader)
        max_iter = self.opt.niter * iter_per_epoch
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Testing', leave=False)

            for iter_num, (raw, labels) in enumerate(progress_bar):
                self.model.eval()
                raw = raw.cuda()
                img_label = np.array(labels)   
                labels = labels.cuda().to(torch.float)
                step = iter_num / max_iter
                outputs, feat_patch, attn_map = self.model(raw, step=step, attn_blk=self.opt.attn_blk, feat_blk=self.opt.feat_blk, k=self.opt.k_weight, thr=self.opt.k_thr)
                output_pred = torch.argmax(F.softmax(outputs, dim=1).cpu().data, dim=1)
                y_pred = np.round(output_pred)
                y = torch.argmax(torch.tensor(img_label), dim=1)
                y_pred = np.array([y_pred])
                y = np.array([y])
                test_acc += accuracy_score(y, y_pred)
                number_steps += 1
                progress_bar.set_postfix({'acc': test_acc/number_steps})
                y_true_list.append(y)
                y_pred_list.append(y_pred)
        
        print("Test Accuracy: ", 100 * accuracy_score(y_true_list, y_pred_list))
        print("Test Precision: ", 100 * precision_score(y_true_list, y_pred_list, labels=[0,1]))
        print("Test Recall: ", 100 * recall_score(y_true_list, y_pred_list, labels=[0,1]))
        print("Test F1 Score: ", 100 * f1_score(y_true_list, y_pred_list, labels=[0,1]))
        y_true_list = [item for sublist in y_true_list for item in sublist]
        y_pred_list = [item for sublist in y_pred_list for item in sublist]
        Abstract.plot_confusion_matrix(y_true_list, y_pred_list, ['Real', 'Fake'], "test")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, type_) -> None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_true, y_pred), display_labels = classes)
        cm_display.plot()
        cm_display.ax_.set_title(type_ + ' Confusion Matrix')
    
class UAIViT(Abstract):
    def __init__(self, opt, num_classes:int = 2, epochs: int = 10, batch_size:int = 128, save_wieghts:bool = False, load_weights:str = None) -> None:
        self.model = vit_base_patch16_224(pretrained=False, num_classes=num_classes).to(device)

        super().__init__(self.model, epochs, batch_size, save_wieghts, load_weights, opt)

    def __call__(self) -> None:
        self.train()
        self.test()