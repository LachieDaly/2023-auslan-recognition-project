"""
PyTorch Lightning module definition
Delegates computation to one of the defined networks (vtn.py, vtn_hc.py, vtn_hcpf.py)
"""

from argparse import ArgumentParser

import lightning as pl # gotta work on this
import torchmetrics
import torch 
from torch.optim.lr_scheduler import StepLR

from .vtn_hcpf import VTNHCPF
from .vtn_hc import VTNHC
from .vtn_fb import VTNFB
from .vtn_rastgoo import VTNR
from .lstm_fb import LSTMFB
from .lstm_hc import LSTMHC
from .lstm_hcpf import LSTMHCPF
from .lstm_rastgoo import LSTMR
from .lstm_hcp import LSTMHCP

def get_model_def():
    return Module

def get_model(**kwargs):
    return Module(**kwargs)

class Module(pl.LightningModule):
    def __init__(self, model, **kwargs):
        """
        Initialises Module being used in training or predicting

        :param model: the string name of the model to be chosen
        """
        super().__init__()

        self.save_hyperparameters()
        NUM_CLASSES = 29
        
        if model == 'vtnhcpf':
            self.model = VTNHCPF(NUM_CLASSES, 
                                self.hparams.num_heads, 
                                self.hparams.num_layers, 
                                self.hparams.embed_size,
                                self.hparams.sequence_length, 
                                self.hparams.cnn,
                                self.hparams.freeze_layers,
                                self.hparams.dropout, 
                                device=self.device)
        elif model == 'vtnhc':
            self.model = VTNHC(NUM_CLASSES,
                               self.hparams.num_heads,
                               self.hparams.num_layers,
                               self.hparams.embed_size,
                               self.hparams.sequence_length,
                               self.hparams.cnn,
                               self.hparams.freeze_layers,
                               self.hparams.dropout,
                               device=self.device)
        elif model == 'vtnfb':
            self.model = VTNFB(NUM_CLASSES,
                               self.hparams.num_heads,
                               self.hparams.num_layers,
                               self.hparams.embed_size,
                               self.hparams.sequence_length,
                               self.hparams.cnn,
                               self.hparams.freeze_layers,
                               self.hparams.dropout,
                               device=self.device)
            
        elif model == 'lstmfb':
            self.model = LSTMFB(NUM_CLASSES,
                                self.hparams.embed_size,
                                self.hparams.sequence_length,
                                self.hparams.cnn,
                                self.hparams.freeze_layers,
                                self.hparams.dropout,
                                device=self.device)
            
        elif model == 'lstmhc':
            self.model = LSTMHC(NUM_CLASSES,
                                self.hparams.embed_size,
                                self.hparams.sequence_length,
                                self.hparams.cnn,
                                self.hparams.freeze_layers,
                                self.hparams.dropout,
                                device=self.device)
        
        elif model == 'lstmhcpf':
            self.model = LSTMHCPF(NUM_CLASSES,
                                self.hparams.embed_size,
                                self.hparams.sequence_length,
                                self.hparams.cnn,
                                self.hparams.freeze_layers,
                                self.hparams.dropout,
                                device=self.device)
            
        elif model == 'lstmrast':
            self.model = LSTMR(NUM_CLASSES,
                                self.hparams.embed_size,
                                self.hparams.sequence_length,
                                self.hparams.cnn,
                                self.hparams.freeze_layers,
                                self.hparams.dropout,
                                device=self.device)

        elif model == 'vtnrast':
            self.model = VTNR(NUM_CLASSES,
                    self.hparams.num_heads,
                    self.hparams.num_layers,
                    self.hparams.embed_size,
                    self.hparams.sequence_length,
                    self.hparams.cnn,
                    self.hparams.freeze_layers,
                    self.hparams.dropout,
                    device=self.device)

        elif model == 'lstmhcp':
            self.model = LSTMHCP(NUM_CLASSES,
                            self.hparams.embed_size,
                                self.hparams.sequence_length,
                                self.hparams.cnn,
                                self.hparams.freeze_layers,
                                self.hparams.dropout,
                                device=self.device)
            
        self.criterion = torch.nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=29)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Fairly typical training step for lightning model
        """
        # Data, Labels
        x, y = batch
        # Prediction
        z = self.model(x)
        # Calculate loss
        loss = self.criterion(z, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.accuracy(z, y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Fairly typical validation setp for lightning model
        """
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy(z, y))
        return loss
    
    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        
        return {
            "optimizer": optimiser,
            "lr_scheduler": StepLR(optimiser, step_size=self.hparams.lr_step_size, gamma=0.1),
            "monitor": "val_accuracy"
        }
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--embed_size", type=int, default=512)
        parser.add_argument("--cnn", type=str, default="rn18")
        parser.add_argument("--freeze_layers", type=int, default=0,
                            help="Freeze all CNN layers up to this index (default: 0, no frozen layers)")
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--dropout", help="Dropout before MHA and FC", type=float, default=0)
        parser.add_argument("--lr_step_size", type=int, default=5)
        parser.add_argument("--model", type=str, required=True)
        return parser