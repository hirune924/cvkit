import argparse
from distutils.util import strtobool
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
import cv2
from pprint import pprint, pformat
from src.inference import InferenceInterface

from torch.utils.data import Dataset, DataLoader

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
    model = InferenceInterface(model_config_path=args.config_path, model_ckpt_path=args.ckpt_path)
    target_list = glob(args.target_path_regex)
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

    result = pd.DataFrame()
    result['image_path'] = target_list
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