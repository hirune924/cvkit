from typing import Dict, Tuple
import torch
import torch.nn.functional as f
import numpy as np
from omegaconf import OmegaConf
from src.model import build_model
from src.augment import build_augment
from src.utils import load_pytorch_model

class InferenceInterface:
    def __init__(
        self,
        model_config_path: str,
        model_ckpt_path: str,
        device: str = 'cuda',
    ):
        self.config = OmegaConf.load(model_config_path)
        self.model = build_model(self.config)
        self.model = load_pytorch_model(model_ckpt_path, self.model)
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def predict(self, feature_dict : Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict.
        Args:
            feature_dict: {"input": torch.Tensor shape (time),
                           "target": torch.tensor(0),  # 評価には使用しないダミーのラベル}.
        Returns:
            ndarray: The predicted class with shape (N).
            ndarray: The prediction confidence (N, classes).
        """
        if len(feature_dict["input"].size()) == 1:
            feature_dict["input"] = feature_dict["input"].unsqueeze(dim=0).to(self.device)
            feature_dict["target"] = feature_dict["target"].unsqueeze(dim=0).to(self.device)
        else:
            feature_dict["input"] = feature_dict["input"].to(self.device)
            feature_dict["target"] = feature_dict["target"].to(self.device)

        outputs_dict = self.model(feature_dict, is_test=True)
        y_hat = outputs_dict["logit_soft"].detach().cpu().numpy()
        pred = np.argmax(y_hat, axis=1)
        return pred, y_hat

    @torch.inference_mode()
    def predict2(self, feature_dict : Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict.
        Args:
            feature_dict: {"input": torch.Tensor shape (N, time)}.
        Returns:
            ndarray: The predicted class with shape (N).
            ndarray: The prediction confidence (N, classes).
        """
        # 1件ずつ推論
        y_hat = []
        for input in feature_dict["input"]:
            f = {"input": input.unsqueeze(dim=0).to(self.device),
                 "target": torch.tensor(0).unsqueeze(dim=0).to(self.device)  # 評価には使用しないダミーのラベル
                 }
            outputs_dict = self.model(f, is_test=True)
            y_h = outputs_dict["logit_soft"].detach().cpu().numpy()
            y_hat.append(y_h)
        y_hat = np.concatenate(np.array(y_hat)).squeeze()
        pred = np.argmax(y_hat, axis=1)
        return pred, y_hat