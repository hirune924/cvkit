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
from src.utils import load_pytorch_model, load_conf, load_one, wav_pad_trunc
from src.augment import build_augment
from src.model import build_model


conf_dict = {'batch_size': 32,
             'epoch': 30,
             'model_name': 'seresnext26t_32x4d',
             'lr': 3e-4,
             'target_fold': 0,
             'ckpt_pth': None,
             'data_path': '???',
             'classes': '???',
             'output_dir': './output/${model_name}', #補完
             'seed': 2021,
             'trainer': {},
             'debug': False,
             #'debug': True,  # 動作確認用。データ数減らして2epochだけ学習する
             'train_wav_crop_len': 30,  # trainは各ファイルの30秒だけを使う
             'valid_wav_crop_len': 5,  # validは各ファイルの5秒だけを使う
             'sr': 32000,  # evaluate.py, predict.py で使うサンプル周波数。BirdCLEF-2023に合わせて32000で固定
             }
if conf_dict['debug']:
    conf_dict['epoch'] = 2

####################
# Dataset
####################
class CustomDataset(Dataset):
    def __init__(self, dataframe, classes, transform=None,
                 is_train=False, wav_crop_len=5):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels_list = classes
        self.is_train = is_train
        self.wav_crop_len = wav_crop_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fp = row["path"]
        label = row["class_id"]
        sr = row["sr"]

        if self.is_train:
            # trainでは wav_len_sec の長さだけを切り出すランダムクロップで波形切り出す
            wav_len = row["frames"]
            wav_len_sec = wav_len / sr
            max_offset = wav_len_sec - self.wav_crop_len
            max_offset = max(max_offset, 1)
            # ランダムに切り出す開始時刻決める
            offset = np.random.randint(max_offset)
            duration = self.wav_crop_len
        else:
            offset = 0
            duration = None

        # 音声ファイルから信号の波形データロード
        wav = load_one(fp, offset, duration)

        # 信号の波形の長さを サンプリング周波数 x wav_crop_len(秒) にする
        wav = wav_pad_trunc(wav, self.wav_crop_len, sr)

        # 背景の雑音などのaugment
        if self.transform:
            wav = self.transform(samples=wav, sample_rate=sr)

        # 波形信号をtensorにする
        wav_tensor = torch.tensor(wav)

        feature_dict = {
            "input": wav_tensor,
            "target": torch.tensor(label.astype(np.float32)),
        }

        return feature_dict

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

            if self.conf['debug']:
                train_df = train_df.sample(n=500, random_state=0).reset_index(drop=True)
                valid_df = valid_df.sample(n=100, random_state=0).reset_index(drop=True)

            train_transform, valid_transform = build_augment(self.conf)

            self.train_dataset = CustomDataset(train_df, self.conf.classes, transform=train_transform,
                                               is_train=True,
                                               wav_crop_len=self.conf.train_wav_crop_len)
            self.valid_dataset = CustomDataset(valid_df, self.conf.classes, transform=valid_transform,
                                               is_train=False,
                                               wav_crop_len=self.conf.valid_wav_crop_len)

        elif stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)

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
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.hparams.classes))

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x, is_test=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        outputs_dict = self.model(batch, is_test=False)
        loss = outputs_dict["loss"]
        return loss

    def validation_step(self, batch, batch_idx):
        outputs_dict = self.model(batch, is_test=True)
        return {
            "val_loss": outputs_dict["loss"],
            "y": outputs_dict["target"],
            "y_hat": outputs_dict["logit_soft"]
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