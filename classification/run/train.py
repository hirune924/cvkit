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
             'epoch': 100,
             'image_size': 256,
             'model_name': 'tf_efficientnet_b1_ns',
             'lr': 0.001,
             'target_fold': 0,
             'ckpt_pth': None,
             'data_path': '???',
             'classes': '???',
             'output_dir': './output/${model_name}', #補完
             'seed': 2021,
             'trainer': {}}

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
        label = self.data.loc[idx, "class_id"]

        return image, label
           
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
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = build_model(conf)
        if self.hparams.ckpt_pth is not None:
            self.model = load_pytorch_model(self.hparams.model_ckpt, self.model)
        self.criteria = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.hparams.classes))

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if torch.rand(1)[0] < 0.5: #self.current_epoch < self.hparams.epoch*1.8:
            # mixup
            alpha = 0.5
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size)
            x = lam * x + (1 - lam) * x[index, :]

            y_hat = self.model(x)
            loss = lam * self.criteria(y_hat, y) + (1 - lam) * self.criteria(y_hat, y[index])
        else:
            y_hat = self.model(x)
            loss = self.criteria(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
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
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_acc', 
                                          save_last=True, save_top_k=1, mode='max', 
                                          save_weights_only=True, filename='{epoch}-{val_acc:.5f}')

    data_module = CustomDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)


if __name__ == "__main__":
    main()