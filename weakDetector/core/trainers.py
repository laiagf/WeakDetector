import torch.nn.functional as F
import torch
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import datetime
from abc import ABC, abstractmethod
from tqdm import tqdm


class Trainer(ABC):

    def __init__(self, model, optimiser, lr, loss_func, lr_decrease_rate=-1, log_interval=200):
        """Construct abstract Trainer class.

        Args:
            model (nn.Module): Model to train
            optimiser (torch.optim): Training optimiser
            lr (Float): Learning rate
            loss_func (function): Loss function
            lr_decrease_rate (int, optional): Learning rate decrease rate. Defaults to 1.
            log_interval (int, optional): Interval of batches to print training loss. Defaults to 200.
        """
        self._model = model

        self._optimiser = optimiser

        self._lr = lr

        self._loss_func = loss_func

        self._lr_decrease_rate = lr_decrease_rate

        self._steps = 0

        self._epoch = 0

        self._log_interval= log_interval

        self._train_losses = []

        self._val_losses = []


    @property 
    def model(self):
        """Get model.
        """
        return self._model
    
    @property 
    @abstractmethod
    def training_log(self):
        """Get dataframe that logs training process.
        """
        pass


    @abstractmethod
    def _batch_loss(self, batch, device):
        """Compute loss for one batch.

        Args:
            batch (tuple): Batch of dataloader
            device (torch.device): Device to perform computations on
        """
        pass
    
    def _train_epoch(self, train_loader, device):
        """Train model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): Training dataloader
            device (torch.device): Device to train on
        """
        print('Train epoch')
        train_loss = 0
        self._model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            #print('batch idx', batch_idx)
            self._optimiser.zero_grad()
            loss = self._batch_loss(batch, device)
            loss.backward()
            #print('loss backward')
            self._optimiser.step()
            train_loss += loss
            self._steps += train_loader.batch_size
            #print('steps done')
            if batch_idx > 0 and batch_idx % self._log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    self._epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item()/self._log_interval, self._steps))
            
        self._train_losses.append(train_loss.cpu().detach().numpy()/len(train_loader.dataset))



    @abstractmethod
    def _val_epoch(self, val_loader, device):
        """Run model on validation set and get metrics for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): Validation dataloader
            device (torch.device): Device to run operations on
        """
        pass
         

        
    def __call__(self, train_loader, val_loader, n_epochs, device, outpath=None, checkpoints_every=-1):
        """Perform training loop.

        Args:
            train_loader (torch.utils.data.Dataloader): Train dataloader
            val_loader (torch.utils.data.Dataloader): Validation dataloader
            n_epochs (int): Number of epochs to train for
            device (torch.device): Device to train and validate on
        """
        print('Starting training')
        self._model.to(device)
        print(device)

        for epoch in range(1, n_epochs+1):
            print(epoch)
            self._train_epoch(train_loader, device)
            self._val_epoch(val_loader, device)
            self._epoch +=1
            if (self._lr_decrease_rate>1)and (epoch % self._lr_decrease_rate == 0):
                self._lr /= 10#self._lr_decrease_rate
                for param_group in self._optimiser.param_groups:
                    param_group['lr'] = self._lr
            
            if epoch % checkpoints_every == 0 and outpath is not None:
                torch.save(self._model.state_dict(), os.path.join(outpath, f'checkpoint_step_{epoch}.pth'))	

        
        #self._summarize_training()


class AETrainer(Trainer):
    def __init__(self, model, optimiser, lr, loss_func=torch.nn.MSELoss(), lr_decrease_rate=-1, log_interval=2000):

        super().__init__(model, optimiser, lr, loss_func, lr_decrease_rate, log_interval)
        
    @property 
    def training_log(self):
        """Get dataframe that logs training process.
        """
        return pd.DataFrame({'epoch':[i for i in range(self._epoch)], 
                           'train_loss': self._train_losses,
                            'val_loss': self._val_losses})

    def _batch_loss(self, batch, device):
        """Compute loss of dataloader batch

        Args:
            batch (tuple): DataLoader batch # TODO IS THIS TUPLE OR NOT
            device (torch.device): Device to perform operations on

        Returns:
            torch.Tensor: Loss value for batch
        """
        batch = batch.to(device)
        _, decoded = self._model(batch)

        return self._loss_func(decoded, batch)
    
    
    def _val_epoch(self, val_loader, device):
        """Run model on validation set and get metrics for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): Validation DataLoader
            device (torch.device): Device to perform operations on
        """
        self._model.eval()
        running_val_loss = 0
        # Iterate through val_loader 
        for click_batch in val_loader:
            # Get decoded data for batch
            _, decoded = self._model(click_batch.to(device))
            # Compute batch loss
            loss_ae = self._loss_func(decoded.to(device), click_batch.to(device))
            
            #Update running_val_loss
            total_loss = loss_ae
            running_val_loss += total_loss.item()*len(click_batch)

        # Print epoch's metrics 
        running_val_loss=running_val_loss/len(val_loader.dataset)
        self._val_losses.append(running_val_loss)
        print(f'{datetime.datetime.now().time().replace(microsecond=0)} %%'
            f'Epoch: {self._epoch}\n'
            f'Training average loss: {self._train_losses[-1]:.4f}\t'
            f'Validation average loss: {running_val_loss:.4f}\n')
        return
    

class ClassifierTrainer(Trainer):

    def __init__(self, model, optimiser, lr, loss_func=F.nll_loss, lr_decrease_rate=10, log_interval=2000):

        super().__init__(model, optimiser, lr,loss_func, lr_decrease_rate, log_interval)
        
        self._val_accuracies = []
        self._val_fscores = []
        self._val_precisions = []
        self._val_recalls = []
    
    
    @property 
    def training_log(self):
        """Get dataframe that logs training process.
        """

        return pd.DataFrame({'epoch':[i for i in range(self._epoch)], 
                            'train_loss': self._train_losses,
                            'val_loss': self._val_losses,
                            'val_accuracy': self._val_accuracies,
                            'val_fscore':self._val_fscores,
                            'val_precision': self._val_precisions,
                            'val_recalls': self._val_recalls })
    

    def _batch_loss(self, batch, device):
        """Compute loss of DataLoader batch

        Args:
            batch (tuple): DataLoader batch
            device (torch.device): Device to perform operations on
        """
        # Separate batch into data and labels (target)
        (data, target) = batch

        #Send data to device
        data = data.to(device)
        target = target.to(device)


        # Run model on data
        output = self._model(data.float())
        # Return loss for batch
        return self._loss_func(output, target)
    
    
    def _val_epoch(self, val_loader, device):
        """Run model on validation set and get metrics for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): Validation DataLoader
            device (torch.device): Device to perform operations on
        """

        self._model.eval()

        val_loss = 0
        correct = 0

        outputs = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader: # Iterate over val_loader
                # Send data and labels to device, and reshape data
                data = data.to(device)
                target = target.to(device)
           #     data = data.view(-1, data.shape[0], data.shape[1])

                # Get model outputs
                output = self._model(data)
                # Add batch loss to epoch loss 
                val_loss += self._loss_func(output, target, size_average=False).item()
                # Compute number of correct predictions
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # Add outputs and targets to running counters
                outputs.append(pred.cpu())
                targets.append(target.data.view_as(pred).cpu())			

            # Compute batch performance metrics
            val_loss /= len(val_loader.dataset)

            val_acc = 100. * correct / len(val_loader.dataset)
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)

            f1 = f1_score(y_pred=outputs,y_true=targets, average='weighted')
            precision = precision_score(y_pred=outputs, y_true=targets, average='weighted')
            recall = recall_score(y_pred=outputs, y_true=targets, average='weighted')

            self._val_accuracies.append(val_acc)
            self._val_fscores.append(f1)
            self._val_recalls.append(recall)
            self._val_precisions.append(precision)
            self._val_losses.append(val_loss)
            # print epoch metrics
            print(f'{datetime.datetime.now().time().replace(microsecond=0)} %% '
            f'Epoch: {self._epoch}\n'
            f'Training average loss: {self._train_losses[-1]:.4f}\t'                  
            f'Validation average loss :{val_loss} \n'
            f'Accuracy: {correct}/{len(val_loader.dataset)} {val_acc}\n'
            f'F1: {f1} \t Recall:{recall} \t Precision: {precision}\n')

            
        return 