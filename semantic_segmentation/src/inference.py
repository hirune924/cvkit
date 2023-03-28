from typing import Tuple
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
        _, self.valid_transform = build_augment(self.config)
        self.device = device

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image.
        Args:
            image (ndarray): The ndarray image with shape (H,W,C).
        Returns:
            torch.Tensor: The transformed torch.Tensor Image with shape (C,H,W)
        """
        image = self.valid_transform(image=image)
        image = torch.from_numpy(image["image"].transpose(2, 0, 1))

        return image

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Predict.
        Args:
            image (torch.Tensor): The torch.Tensor image with shape (C,H,W) or (N,C,H,W).
        Returns:
            ndarray: The prediction with shape (N, C, H, W).
        """
        if len(image.size()) == 3:
            image = image.unsqueeze(dim=0)

        pred = torch.sigmoid(self.model(image.to(self.device))).detach().cpu().numpy()
        return pred