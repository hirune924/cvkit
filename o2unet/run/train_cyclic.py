import os
import sys
import cv2
import numpy as np
import pandas as pd
from pprint import pprint, pformat
from loguru import logger

import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

import glob
from sklearn.metrics import roc_auc_score
from src.utils import load_pytorch_model, load_conf
from src.augment import build_augment
from src.model import build_model


conf_dict = {'batch_size': 32, 
             'num_cycle': 5,
             'image_size': 256,
             'model_name': 'tf_efficientnet_b1_ns',
             'lr': 0.001,
             'target_fold': 0,
             'ckpt_pth': '???',
             'data_path': '???',
             'classes': '???',
             'output_dir': '/output/${model_name}/o2u', #補完
             'seed': 2021,
             'trainer': {}}

####################
# Utils
####################
class SampleValuesLogger():
    def __init__(self, file_name, base_df):
        self.file_name = file_name
        self.base_df = base_df
        base_df.to_csv(file_name, index=False)
        
    def log(self, values, index, col_name):
        df = pd.DataFrame(data = {col_name: values}, index=index)
        base_df = pd.read_csv(self.file_name)
        base_df.join(df).to_csv(self.file_name, index=False)
        
####################
# Dataset
####################
class CustomDataset(Dataset):
    def __init__(self, dataframe, classes, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels_list = classes
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "image_path"]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = torch.from_numpy(image["image"].transpose(2, 0, 1))
        label = self.data.loc[idx, "class_id"].astype(int)

        return image, label, idx
           
####################
# Data Module
####################

class CustomDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf       

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(self.conf.data_path)
            train_df = df[df['fold']!=self.conf.target_fold]
            valid_df = df[df['fold']==self.conf.target_fold]
            
            train_transform, valid_transform = build_augment(self.conf)
            
            self.train_dataset = CustomDataset(train_df, self.conf.classes, transform=train_transform)
            self.valid_dataset = CustomDataset(valid_df, self.conf.classes, transform=valid_transform)
            
        elif stage == 'test':
            pass
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        pass

####################
# Lightning Module
####################
class LitSystem(pl.LightningModule):
    def __init__(self, conf, loss_logger):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = build_model(conf)
        if self.hparams.ckpt_pth is not None:
            self.model = load_pytorch_model(self.hparams.ckpt_pth, self.model)
        self.criteria = torch.nn.CrossEntropyLoss(reduction='none')
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.hparams.classes))
        self.loss_logger = loss_logger

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams.lr*0.05, 
        max_lr=self.hparams.lr, step_size_up=5, step_size_down=5, mode='triangular', cycle_momentum=False)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return { "loss": loss.mean(), "losses": loss, "index": idx}

    def training_epoch_end(self, outputs):
        losses = torch.cat([x["losses"] for x in outputs]).cpu().detach().numpy()
        index = torch.cat([x["index"] for x in outputs]).cpu().detach().numpy()
        self.loss_logger.log(losses, index, f'loss_{self.current_epoch}')
    
    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y).mean()
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu()

        preds = np.argmax(y_hat, axis=1)
        val_acc = self.accuracy(y, preds)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_acc', val_acc)

def main():
    conf = load_conf(base_conf=conf_dict)
    logger.info('Conf:\n'+pformat(OmegaConf.to_container(conf, resolve=True)))
    seed_everything(conf.seed)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))
    lr_monitor = LearningRateMonitor(logging_interval='step')

    data_module = CustomDataModule(conf)
    data_module.setup(stage='fit')

    loss_logger = SampleValuesLogger(file_name=os.path.join(conf.output_dir, 'o2u.csv'), base_df=data_module.train_dataset.data)

    lit_model = LitSystem(conf, loss_logger=loss_logger)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor],
        max_epochs=conf.num_cycle*10,
        gpus=-1,
        amp_backend='native',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=False,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

    ## Calculate O2U score
    o2u_log = pd.read_csv(os.path.join(conf.output_dir, 'o2u.csv'))
    normalize_loss_df = o2u_log.iloc[:, o2u_log.columns.str.startswith('loss_')] - o2u_log.iloc[:, o2u_log.columns.str.startswith('loss_')].mean()
    o2u_log['loss_avg'] = normalize_loss_df.mean(axis='columns')
    o2u_log['loss_var'] = normalize_loss_df.var(axis='columns')
    o2u_log.to_csv(os.path.join(conf.output_dir, 'o2u.csv'), index=False)
    
if __name__ == "__main__":
    main()