import timm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import torchaudio as ta

from src import audio_augments
from src.augment import random_low_pass_filter_torch


def build_model(conf):
    return BirdClassifier(model_name=conf.model_name, num_classes=len(conf.classes), pretrained=True, in_chans=1)


class ConfigMelSpec:
    def __init__(self):
        # MelSpectrogram
        self.sample_rate = 32000
        self.n_fft = 1024
        self.window_size = 1024
        self.hop_length = 320
        self.f_min = 0
        self.f_max = 14000
        self.pad = 0
        self.mel_bins = 128
        self.power = 2
        self.normalized = False
        # AmplitudeToDB
        self.top_db = None
        # MelSpectrogram Augments
        self.enable_gaussnoise = True
        self.enable_masking = True
        self.mel_mixup = 0.5
        self.mel_mixup2 = 0.5
        # train duration sec
        self.wav_crop_len = 30

class BirdClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained, in_chans):
        super().__init__()

        cfg = ConfigMelSpec()
        self.cfg = cfg

        self.num_classes = num_classes

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.window_size,
            win_length=cfg.window_size,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            pad=cfg.pad,
            n_mels=cfg.mel_bins,
            power=cfg.power,
            normalized=cfg.normalized,
        )
        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

        self.enable_gaussnoise = cfg.enable_gaussnoise
        if self.enable_gaussnoise:
            self.gaussnoise = audio_augments.GaussianNoiseSNRTorch(min_snr=5, max_snr=20, p=0.5)

        self.enable_masking = cfg.enable_masking
        if self.enable_masking:
            self.freq_mask = ta.transforms.FrequencyMasking(24, iid_masks=True)
            self.time_mask = ta.transforms.TimeMasking(64, iid_masks=True)

        base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_chans,
            drop_rate=0.5,
            drop_path_rate=0.5,
         )
        if "efficientnet" in model_name:
            backbone_out = base_model.num_features
        else:
            backbone_out = base_model.feature_info[-1]["num_chs"]

        self.backbone_out = backbone_out
        self.backbone = base_model

        self.gem = GeM(p=3, eps=1e-6)

        # 30 seconds -> 5 seconds
        wav_crop_len = cfg.wav_crop_len
        self.factor = int(wav_crop_len / 5.0)

        # https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/aec59636e1e87b52e08747b16a9e083f0ab0421f/models/ps_model_9.py
        self.mixup = Mixup(mix_beta=1)

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # nn.CrossEntropyLoss()

        self.head1 = nn.Linear(backbone_out, num_classes, bias=True)

    def forward(self, batch, is_test=False):
        x = batch["input"]
        y = batch["target"]

        # Convert to one-hot
        y_one_hot = nn.functional.one_hot(y.long(), num_classes=self.num_classes).float()

        if is_test == False:
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)  # 30 -> 5 secにするためbatchにデータ渡す

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)   # bs, n_mels(=freq), time
            x = (x + 80) / 80  # top_dbの80デシベルで割って正規化

            # spectrogramsに負の数があるとtorch.powがnanになるので絶対値とる
            x = torch.abs(x)
            # 2乗してspectrogramsのコントラスト強める
            x = torch.pow(x, 2)

        # MelSpec画像にGaussianNoise
        if self.training and self.enable_gaussnoise:
            x = self.gaussnoise(x)

        # MelSpec画像にmask
        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # MelSpec画像にrandom_low_pass_filter
        if self.training and (np.random.rand() <= 0.5):
            x = x.permute(0, 2, 1)  # bs, time, n_mels(=freq)
            # random_low_pass_filter_torch()はbs, time, n_mels(=freq)のshapeじゃないとだめ
            x = random_low_pass_filter_torch(x)
            x = x.permute(0, 2, 1)  # bs, n_mels(=freq), time

        x = x.permute(0, 2, 1)  # bs, time, n_mels(=freq)
        x = x[:, None, :, :]  # bs, ch, time, n_mels(=freq)

        # 5sec単位で分割したMelSpec画像をmixup
        # https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/aec59636e1e87b52e08747b16a9e083f0ab0421f/models/ps_model_9.py
        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if np.random.random() <= self.cfg.mel_mixup:
                x, y_one_hot = self.mixup(x, y_one_hot)
            if np.random.random() <= self.cfg.mel_mixup2:
                x, y_one_hot = self.mixup(x, y_one_hot)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)

        # MelSpec画像をtimmのモデルでforward
        x = self.backbone(x)

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)  # b, t, c, f
            x = x.reshape(b // self.factor, self.factor * t, c, f)  # 30 -> 5 secにしたbatchをtimeにもどす
            x = x.permute(0, 2, 1, 3)  # b, c, t, f

        x = self.gem(x) # b, c, 1, 1

        x = x[:, :, 0, 0]

        logit = self.head1(x)

        # Model内でloss出す
        loss = self.loss_fn(logit, y_one_hot)
        loss = loss.sum()

        return {"loss": loss,
                "logit": logit,
                "logit_soft": torch.softmax(logit, 1),
                "target": y,
                }

####################
# https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/main/models/model_utils.py
####################
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )

class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight