import torch
import os
from abc import ABC
from pathlib import Path
from tqdm import tqdm
import inspect
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import os
from pathlib import Path
import inspect
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
from typing import Any
import pandas as pd
from sklearn.model_selection import KFold
from utils import AverageMeter, FocalLossWithWeights, FocalLoss
from data_loader import data_loader
from custom_model import CustomRain



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

    def __init__(self, model: nn.modules, epochs:int = 10, batch_size:int = 128, save_wieghts:bool = False,
                load_weights:str = None, opt: Any = None, lr: float = 1e-4, scheduler_lr: float = 1e-6):
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
    

    def validate(self, loss_fn: nn.Module, val_loader: DataLoader, autocast: torch.cuda.amp.autocast) -> tuple[float, float]:
        """
        Validate the model on the validation set.

        Args:
            loss_fn (nn.Module): The loss function.
            val_loader (DataLoader): The validation data loader.

        Returns:
            tuple[float, float]: The mean loss and validation accuracy.
        """
        self.model.eval()
        with torch.no_grad():
            losses = AverageMeter() 
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for (sagittal_T2_l1_l2, axial_l1_l2, sagittal_T2_l2_l3,
                 axial_l2_l3, sagittal_T2_l3_l4, axial_l3_l4, sagittal_T2_l4_l5, axial_l4_l5, sagittal_T2_l5_s1,
                   axial_l5_s1, reordered_labels) in progress_bar:
 
                # sagittal_stack = sagittal_stack.cuda()
                sagittal_T2_l1_l2 = sagittal_T2_l1_l2.cuda()
                axial_l1_l2 = axial_l1_l2.cuda()
                sagittal_T2_l2_l3 = sagittal_T2_l2_l3.cuda()
                axial_l2_l3 = axial_l2_l3.cuda()
                sagittal_T2_l3_l4 = sagittal_T2_l3_l4.cuda()
                axial_l3_l4 = axial_l3_l4.cuda()
                sagittal_T2_l4_l5 = sagittal_T2_l4_l5.cuda()
                axial_l4_l5 = axial_l4_l5.cuda()
                sagittal_T2_l5_s1 = sagittal_T2_l5_s1.cuda()
                axial_l5_s1 = axial_l5_s1.cuda()
                labels = reordered_labels.cuda().to(torch.long)
    
                loss_dis = 0.0

                with autocast:
                    outputs = self.model(axial_l1_l2, sagittal_T2_l1_l2, axial_l2_l3,
                                             sagittal_T2_l2_l3, axial_l3_l4, sagittal_T2_l3_l4, axial_l4_l5, sagittal_T2_l4_l5,
                                               axial_l5_s1, sagittal_T2_l5_s1)

                    
                    for col in range(25):
                        pred = outputs[:,col*3:col*3+3]
                        gt = labels[:,col]
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        print("Number of trainable parameters:", Abstract.count_parameters(self.model))
        scheduler = Abstract.get_scheduler_reduceOnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001)


        weights = torch.tensor([1.0, 2.0, 4.0], dtype= torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        criterion = criterion.cuda()
        loss_fn, weights = [], []
        for loss_name, weight in {criterion:1}.items():
            loss_fn.append(loss_name.cuda())
            weights.append(weight)
        # summary_writer = SummaryWriter(self.name + "_logs" + str(time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())))
        train_losses = []
        
        train_labels = r"F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\csv\train_series_descriptions_with_paths.csv"
        train_path = r"F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\train.csv"
        train_descriptions = r"F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\train_series_descriptions.csv"
        train_dataset = data_loader(train_path, train_labels, train_descriptions)
        validation_dataset = data_loader(train_path, train_labels, train_descriptions, mode = 'val')
        train_df = pd.read_csv(train_path)
        primary_labels = train_df['study_id'].values
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(primary_labels)))):
            print(f"Fold: {fold + 1}")
            best_valid_loss = np.inf
            early_stopping_counter = 0
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle = True, pin_memory=True, num_workers=2)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)
            # validation_loader_2 = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

            for current_epoch in range(self.epochs):
                print('Epoch {}/{}'.format(current_epoch+1, self.epochs))
                running_loss = 0.0
                progress_bar = tqdm(train_loader, desc='Training', leave=False)
                losses = AverageMeter() 
                for (sagittal_T2_l1_l2, axial_l1_l2, sagittal_T2_l2_l3,
                 axial_l2_l3, sagittal_T2_l3_l4, axial_l3_l4, sagittal_T2_l4_l5, axial_l4_l5, sagittal_T2_l5_s1,
                   axial_l5_s1, reordered_labels) in progress_bar:
 
                    self.model.train(True)
                    # sagittal_stack = sagittal_stack.cuda()
                    sagittal_T2_l1_l2 = sagittal_T2_l1_l2.cuda()
                    axial_l1_l2 = axial_l1_l2.cuda()
                    sagittal_T2_l2_l3 = sagittal_T2_l2_l3.cuda()
                    axial_l2_l3 = axial_l2_l3.cuda()
                    sagittal_T2_l3_l4 = sagittal_T2_l3_l4.cuda()
                    axial_l3_l4 = axial_l3_l4.cuda()
                    sagittal_T2_l4_l5 = sagittal_T2_l4_l5.cuda()
                    axial_l4_l5 = axial_l4_l5.cuda()
                    sagittal_T2_l5_s1 = sagittal_T2_l5_s1.cuda()
                    axial_l5_s1 = axial_l5_s1.cuda()
                    labels = reordered_labels.cuda().to(torch.long)

                    
                    loss = 0.0

                    with autocast:
                        # outputs = outputs.reshape(-1, 3, 25)
                        optimizer.zero_grad()
                        output = self.model(axial_l1_l2, sagittal_T2_l1_l2, axial_l2_l3,
                                             sagittal_T2_l2_l3, axial_l3_l4, sagittal_T2_l3_l4, axial_l4_l5, sagittal_T2_l4_l5,
                                               axial_l5_s1, sagittal_T2_l5_s1)
                        for col in range(25):
                            pred = output[:,col*3:col*3+3]
                            gt = labels[:,col]
                            loss = loss + criterion(pred, gt) / 25
                        loss.backward()
                        # scaler.unscale_(optimizer_l1_l2)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    loss_dis_total = loss
                    
                    running_loss += loss_dis_total.item()
                    losses.update(loss_dis_total.item())
                    progress_bar.set_postfix({'epoch': current_epoch, 'loss': losses.avg})
                train_losses.append(running_loss / (len(train_loader) * self.batch_size))
                valid_loss = self.validate(criterion, val_loader, autocast)
                # validation_2 = self.validate(criterion, validation_loader_2, autocast)
                scheduler.step(valid_loss)
                # scheduler_l2_l3.step(valid_loss)
                # scheduler_l3_l4.step(valid_loss)
                # scheduler_l4_l5.step(valid_loss)
                # scheduler_l5_s1.step(valid_loss)
                

                print("Last LR", scheduler.get_last_lr())
                print('Train Loss: {:.4f}'.format(losses.avg))
                print('--------------------------------')
                # Check for overfitting
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.Validation_loss = best_valid_loss
                    early_stopping_counter = 0
                    
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= 2:  # Stop if validation loss doesn't improve for 2 epochs
                    print("Early stopping due to overfitting")
                    break
                save_path = r"F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification"
                if self.save_wieghts and (early_stopping_counter == 0 or valid_loss < best_valid_loss):
                    self.save_model(self.model, os.path.join(save_path ,self.name + "_best_validation.pt"))
                    # self.save_model(self.model_l1_l2, os.path.join(save_path ,self.name + "_best_validation_l1_l2.pt"))
                    # self.save_model(self.model_l2_l3, os.path.join(save_path ,self.name + "_best_validation_l2_l3.pt"))
                    # self.save_model(self.model_l3_l4, os.path.join(save_path ,self.name + "_best_validation_l3_l4.pt"))
                    # self.save_model(self.model_l4_l5, os.path.join(save_path ,self.name + "_best_validation_l4_l5.pt"))
                    # self.save_model(self.model_l5_s1, os.path.join(save_path ,self.name + "_best_validation_l5_s1.pt"))
            
            if self.save_wieghts:
                old_path = os.path.join(save_path, self.name + "_best_validation.pt")
                new_path = os.path.join(save_path, self.name + f"_{self.Validation_loss}_fold_{fold+1}.pt")
                os.rename(old_path, new_path)
                # self.save_model(self.model_l1_l2, os.path.join(save_path ,self.name + "_best_validation.pt"))
                # models = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
                # for model in models:
                #     old_path = os.path.join(save_path, self.name + f"_best_validation_{model}.pt")
                #     new_path = os.path.join(save_path, self.name + f"_{model}_{self.Validation_loss}_fold_{fold+1}.pt")
                #     os.rename(old_path, new_path)
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, type_) -> None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_true, y_pred), display_labels = classes)
        cm_display.plot()
        cm_display.ax_.set_title(type_ + ' Confusion Matrix')

class RainDrop(Abstract):
    def __init__(self, num_classes =75, epochs: int = 10, batch_size:int = 128, save_wieghts:bool = False, load_weights:str = None) -> None:
        self.model = CustomRain(num_classes, True).to(device)
        # self.model_l2_l3 = CustomRain(num_classes, True).to(device)
        # self.model_l3_l4 = CustomRain(num_classes, True).to(device)
        # self.model_l4_l5 = CustomRain(num_classes, True).to(device)
        # self.model_l5_s1 = CustomRain(num_classes, True).to(device)

        super().__init__(self.model, epochs, batch_size, save_wieghts, load_weights, lr=3.6e-5)

    def __call__(self) -> None:
        self.train()


if __name__ == "__main__":
    model = RainDrop(num_classes=15, epochs=5, batch_size=8, save_wieghts=True)
    model()