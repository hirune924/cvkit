import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from pprint import pprint, pformat
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

CLASSES = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 
'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 
'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 
'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 
'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 
'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 
'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 
'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 
'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 
'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 
'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 
'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 
'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 
'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 
'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 
'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 
'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 
'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 
'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']

def main(args):
    logger.info('\nargs:\n'+pformat(vars(args)))

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / 'labels.csv')
    df['image_path'] = data_dir / 'train' 
    df['image_path'] = df['image_path'] / (df['id'] + '.jpg')
    df['label'] = df['breed'].map(dict(zip(CLASSES, list(range(len(CLASSES))))))

    logger.info('CV split')
    kf = StratifiedKFold(n_splits=args.num_fold, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(df.values, df['label'])):
        df.loc[val_index, "fold"] = int(fold)
    df["fold"] = df["fold"].astype(int)

    logger.info('Save')
    df = df[['image_path', 'label', 'fold']]
    df.to_csv(args.output_path, index=False)
    
    logger.info('\n'+pformat(df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_fold', type=int, default=5)
    args = parser.parse_args()
    main(args)