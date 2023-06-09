from omegaconf import OmegaConf
import datetime
import os
import torch
import sys
import shutil
from loguru import logger

import librosa
import numpy as np
import pandas as pd
from sklearn import metrics
from soundfile import SoundFile


####################
# Utils
####################
def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    res = model.load_state_dict(new_state_dict, strict=False)
    logger.info(res)
    return model

def load_conf(base_conf, include_exex_info=True, save_conf=True, save_code=True):
    cli_conf = OmegaConf.from_cli()
    if cli_conf.get('config',None) is not None:
        override_conf = OmegaConf.load(cli_conf.pop('config'))
    else:
        override_conf = OmegaConf.create()
    base_conf = OmegaConf.create(base_conf)
    conf =  OmegaConf.merge(base_conf, override_conf, cli_conf)

    if include_exex_info:
        exec_info = OmegaConf.create({'exec_info':{'script':sys.argv[0], 'time': str(datetime.datetime.today())}})
    conf = OmegaConf.merge(conf, exec_info)

    os.makedirs(conf.output_dir, exist_ok=True)
    if save_conf:
        OmegaConf.save(config=conf,f=os.path.join(conf.output_dir, 'config.yml'))
    if save_code:
        shutil.copy(sys.modules['__main__'].__file__, os.path.join(conf.output_dir, 'main.py'))

    return conf


####################
# Sound Utils
####################
def load_one(fp, offset=0.0, duration=None):
    """音声ファイルロード"""
    try:
        wav, sr = librosa.load(fp, sr=None, offset=offset, duration=duration)
    except:
        print("FAIL READING rec", fp)
    return wav

def get_audio_info(filepath):
    """Get some properties from an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate  # サンプリング周波数
        frames = f.frames  # 何点分のデータが入っている
        duration = float(frames)/sr  # 音声ファイルの時間sec
    return {"frames": frames, "sr": sr, "duration": duration}

def wav_pad_trunc(wav, wav_crop_len, sample_rate):
    """音声信号の長さを埋める（padding）/ 捨てる(truncate)"""
    # 信号の波形の長さを サンプリング周波数 x wav_crop_len(秒) にする
    expected_length = wav_crop_len * sample_rate
    if wav.shape[0] < expected_length:
        # 時間足りない場合はpadding
        pad = wav_crop_len * sample_rate - wav.shape[0]
        wav_orig = wav.copy()
        l = wav.shape[0]
        if pad >= l:
            while wav.shape[0] <= expected_length:
                wav = np.concatenate([wav, wav_orig], axis=0)
        else:
            max_offset = l - pad
            offset = np.random.randint(max_offset)
            wav = np.concatenate([wav, wav_orig[offset : offset + pad]], axis=0)
    return wav[:expected_length]





