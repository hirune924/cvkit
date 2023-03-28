import os
import sys
import cv2
import numpy as np
import pandas as pd
from pprint import pprint, pformat
from loguru import logger

import pytorch_lightning as pl
from torchmetrics import Dice
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn import model_selection
import albumentations as A
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf

import glob
from src.utils import load_pytorch_model, load_conf
from src.augment import build_augment, rand_bbox
from src.model import build_model


conf_dict = {'batch_size': 32, 
             'epoch': 100,
             'image_size': 512,
             'augment': 'v1',
             'model_name': 'unet>efficientnet-b0',
             'lr': 0.001,
             'target_fold': 0,
             'ckpt_pth': None,
             'data_path': '???',
             'output_dir': './output/${model_name}', #補完
             'seed': 2023,
             'trainer': {}}

####################
# Dataset
####################
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "image_path"]
        mask_name = self.data.loc[idx, "mask_path"]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_name)

        trans = self.transform(image=image, mask=mask)
        image = torch.from_numpy(trans["image"].transpose(2, 0, 1))
        mask = torch.from_numpy(trans["mask"]).unsqueeze(dim=0).float()

        return image, mask
                      
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
            
            self.train_dataset = CustomDataset(train_df, transform=train_transform)
            self.valid_dataset = CustomDataset(valid_df, transform=valid_transform)
            
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
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        self.diceloss = smp.losses.DiceLoss(mode='binary')
        self.score = Dice(average='samples', threshold=0.5)
        self.score_micro = Dice(average='micro', threshold=0.5)

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        lam = np.random.beta(0.5, 0.5)
        rand_index = torch.randperm(x.size()[0]).type_as(x).long()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        y[:, :, bbx1:bbx2, bby1:bby2] = y[rand_index, :, bbx1:bbx2, bby1:bby2]

        y_hat = self.model(x)
        loss = self.bceloss(F.logsigmoid(y_hat).exp(), y) + self.diceloss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.bceloss(F.logsigmoid(y_hat).exp(), y) + self.diceloss(y_hat, y)
        score = self.score(F.logsigmoid(y_hat).exp(), y.long())
        score_micro = self.score_micro(F.logsigmoid(y_hat).exp(), y.long())
        
        return {
            "val_loss": loss,
            "val_dice": score,
            "val_dice_micro": score_micro
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_dice = torch.stack([x["val_dice"] for x in outputs]).mean()
        avg_val_dice_micro = torch.stack([x["val_dice_micro"] for x in outputs]).mean()

        self.log('val_loss', avg_val_loss)
        self.log('val_dice', avg_val_dice)
        self.log('val_dice_micro', avg_val_dice_micro)

def main():
    conf = load_conf(base_conf=conf_dict)
    logger.info('Conf:\n'+pformat(OmegaConf.to_container(conf, resolve=True)))
    seed_everything(conf.seed)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_dice', 
                                          save_last=True, save_top_k=1, mode='max', 
                                          save_weights_only=True, filename='{epoch}-{val_dice:.5f}')

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