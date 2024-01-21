import numpy as np
import torch
from pathlib import Path

# code based on:
# https://github.com/Bjarten/early-stopping-pytorch/tree/master


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, dir, dataset_name, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            dir (str): Directory for the checkpoint to be saved
            dataset_name (str): Name of the dataset for the model
                            Base and surrogate_model will be saved to 
                            path = "dir/dataset_name/{base/surrogate}_model_checkpoint.pt"
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.dataset_name = dataset_name
        self.dir = dir
        
        self._create_directories()
    
    def _create_directories(self):
        path = Path(f"{self.dir}/{self.dataset_name}")
        path.mkdir(parents=True, exist_ok=True)
            
    def __call__(self, val_loss, base_model, surrogate_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, base_model, surrogate_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, base_model, surrogate_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, base_model, surrogate_model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            print("Saving models ...")
        torch.save(base_model.state_dict(), f"{self.dir}/{self.dataset_name}/base_model_checkpoint.pt")
        torch.save(surrogate_model.state_dict(), f"{self.dir}/{self.dataset_name}/surrogate_model_checkpoint.pt")
        self.val_loss_min = val_loss
