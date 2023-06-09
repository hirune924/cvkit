import torch
import torch.nn as nn
import audiomentations as AA


def build_augment(conf):
    train_transform = AA.Compose(
            [
                #AA.AddBackgroundNoise(sounds_path=f"{noise_DIR}/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5),  # 背景に雑音信号追加
                AA.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),  # 信号をランダムに時間方向にシフト（移動）
            ]
        )

    valid_transform = None

    return train_transform, valid_transform


# https://github.com/google-research/chirp/blob/d3824e5f032e6f80ecd0fbb7dc06e6958ede6ec8/chirp/audio_utils.py#LL291C1-L333C28
def random_low_pass_filter_torch(
    melspec: torch.Tensor,
    time_axis: int = -2,
    channel_axis: int = -1,
    min_slope: float = 2.0,
    max_slope: float = 8.0,
    min_offset: float = 0.0,
    max_offset: float = 5.0,
) -> torch.Tensor:
    """
    メルスペクトログラムの画像に対してランダムなローパスフィルタ（低周波数だけ通す）を適用
    Args:
        melspec: （batch形式の）メルスペクトログラムの、最後の軸に周波数を持つものと仮定する。shape=(bs,time,freq) or (bs,ch,time,freq)。time,freqの順番でないとだめ
        time_axis: 時間軸
        channel_axis: 周波数軸
        min_slope: ローパスフィルタの最小スロープ（スロープ=低域通過フィルタの傾き）
        max_slope: ローパスフィルタの最大スロープ（スロープ=低域通過フィルタの傾き）
        min_offset: ローパスフィルタのオフセット（信号通さない領域）の最小値
        max_offset: ローパスフィルタのオフセット（信号通さない領域）の最大値
    Returns:
        低域通過フィルタが適用されたメルスペクトログラム
    """

    shape = list(melspec.shape)
    shape[time_axis] = shape[channel_axis] = 1

    slope = torch.rand(shape) * (max_slope - min_slope) + min_slope
    offset = torch.rand(shape) * (max_offset - min_offset) + min_offset

    shape = [1] * melspec.ndim
    shape[channel_axis] = melspec.shape[channel_axis]
    xspace = torch.linspace(0.0, 1.0, melspec.shape[channel_axis])
    xspace = xspace.view(shape)

    envelope = 1 - 0.5 * (torch.tanh(slope * (xspace - 0.5) - offset) + 1)
    return melspec * envelope.to(melspec.device)





