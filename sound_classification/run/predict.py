import argparse
from distutils.util import strtobool
import os
from glob import glob
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from pprint import pprint, pformat
from src.inference import InferenceInterface
from src.utils import load_one, wav_pad_trunc

import torch
from torch.utils.data import Dataset, DataLoader


class InferenceDataset2(Dataset):
    def __init__(self, target, sr=32000, wav_crop_len=5):
        self.target = target
        self.sr = sr  # サンプル周波数
        self.wav_crop_len = wav_crop_len  # validは各ファイルの5秒だけを使う

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        fp = self.target[idx]
        fn = Path(fp).stem
        offset = 0
        duration = None

        # 音声ファイルから信号の波形データロード
        wav = load_one(fp, offset, duration)

        # 音声ファイルの秒数
        wav_sec = len(wav)//self.sr

        # 信号の波形の長さを サンプリング周波数 x self.wav_crop_len  にバラす
        wav_chunk = np.array( [wav[i*self.sr :(i + self.wav_crop_len)*self.sr ] for i in range(0, wav_sec, self.wav_crop_len)] )

        # self.wav_crop_len単位でrow_id作成
        row_ids = np.array( [f"{fn}_{i}" for i in range(self.wav_crop_len, wav_sec+1, self.wav_crop_len)] )

        # 波形信号をtensorにする
        wav_tensor = torch.tensor(wav_chunk)

        feature_dict = {
            "input": wav_tensor,
        }

        return feature_dict, row_ids, [fp]*len(wav_tensor)

def main(args):
    logger.info('\nargs:\n'+pformat(vars(args)))
    config = OmegaConf.load(args.config_path)
    model = InferenceInterface(model_config_path=args.config_path, model_ckpt_path=args.ckpt_path)
    target_list = glob(args.target_path_regex)
    logger.info(f'number of target data is {len(target_list)}')
    dataset = InferenceDataset2(target_list,
                                sr=config.sr,
                                wav_crop_len=config.valid_wav_crop_len)

    preds = []
    confidence = []
    row_ids = []
    paths = []
    for (batch, batch_row_ids, batch_paths) in tqdm(dataset):
        batch_preds, batch_confidence = model.predict2(batch)
        preds.append(batch_preds)
        confidence.append(batch_confidence)
        row_ids.append(batch_row_ids)
        paths.append(batch_paths)
    preds = np.concatenate(preds, axis=0)
    confidence = np.concatenate(confidence, axis=0)
    row_ids = np.concatenate(row_ids, axis=0)
    paths = np.concatenate(paths, axis=0)

    result = pd.DataFrame()
    result['path'] = paths
    result['row_ids'] = row_ids
    result['predict'] = preds
    conf_columns = ['confidence_'+c for c in list(model.config.classes)]
    result = pd.concat([result, pd.DataFrame(confidence, columns=conf_columns)], axis=1)
    logger.info('result dataframe:\n'+str(result))
    result.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('target_path_regex', type=str)
    args = parser.parse_args()
    main(args)