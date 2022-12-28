import argparse
from distutils.util import strtobool
import os
from glob import glob
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
import cv2
from pprint import pprint, pformat
from src.inference import InferenceInterface

from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

class InferenceDataset(Dataset):
    def __init__(self, target, transform):
        self.target = target
        self.transform = transform
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        img_name = self.target[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        return image

def main(args):
    logger.info('\nargs:\n'+pformat(vars(args)))
    config = OmegaConf.load(args.config_path)
    df = pd.read_csv(config.data_path)
    valid_df = df[df['fold']==config.target_fold].reset_index(drop=True)

    model = InferenceInterface(model_config_path=args.config_path, model_ckpt_path=args.ckpt_path)
    target_list = valid_df['image_path'].values
    logger.info(f'number of target data is {len(target_list)}')
    dataset = InferenceDataset(target_list, model.preprocess)
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

    logger.info('metrics:\n' + str(metrics.classification_report(result['class_id'], result['predict'], target_names=config.classes)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    main(args)