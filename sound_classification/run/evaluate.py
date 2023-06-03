import argparse
from distutils.util import strtobool
import os
from glob import glob
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
from pprint import pprint, pformat
from src.inference import InferenceInterface
from src.utils import load_one, wav_pad_trunc

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics


class InferenceDataset(Dataset):
    def __init__(self, target, sr=32000, wav_crop_len=5):
        self.target = target
        self.sr = sr  # サンプル周波数
        self.wav_crop_len = wav_crop_len  # validは各ファイルの5秒だけを使う

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        fp = self.target[idx]
        offset = 0
        duration = None

        # 音声ファイルから信号の波形データロード
        wav = load_one(fp, offset, duration)

        # 信号の波形の長さを サンプリング周波数 x wav_crop_len(秒) にする
        wav = wav_pad_trunc(wav, self.wav_crop_len, self.sr)

        # 波形信号をtensorにする
        wav_tensor = torch.tensor(wav)

        feature_dict = {
            "input": wav_tensor,
            "target": torch.tensor(0),  # 評価には使用しないダミーのラベル
        }

        return feature_dict

def main(args):
    logger.info('\nargs:\n'+pformat(vars(args)))
    config = OmegaConf.load(args.config_path)
    df = pd.read_csv(config.data_path)
    valid_df = df[df['fold']==config.target_fold].reset_index(drop=True)

    model = InferenceInterface(model_config_path=args.config_path, model_ckpt_path=args.ckpt_path)
    target_list = valid_df['path'].values
    logger.info(f'number of target data is {len(target_list)}')
    dataset = InferenceDataset(target_list,
                               sr=config.sr,
                               wav_crop_len=config.valid_wav_crop_len)
    dataloader = DataLoader(dataset, batch_size=model.config.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)

    preds = []
    confidence = []
    for batch in tqdm(dataloader):
        batch_preds, batch_confidence = model.predict(batch)
        preds.append(batch_preds)
        confidence.append(batch_confidence)
    preds = np.concatenate(preds, axis=0)
    confidence = np.concatenate(confidence, axis=0)

    result = valid_df
    result['predict'] = preds
    conf_columns = ['confidence_'+c for c in list(model.config.classes)]
    result = pd.concat([result, pd.DataFrame(confidence, columns=conf_columns)], axis=1)
    logger.info('result dataframe:\n'+str(result))
    result.to_csv(args.output_path, index=False)

    logger.info('metrics:\n' + str(metrics.classification_report(result['class_id'], result['predict'])))  # , target_names=config.classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    main(args)