"""Standalone inference for streaming RF SNN models."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingSNN(nn.Module):
    def __init__(self, width, use_skip):
        super().__init__()

        self.fc1 = nn.Linear(64, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)

        self.skip = (
            nn.Linear(64, 1, bias=False)
            if use_skip else None
        )

    def forward(self, x):
        """
        x: [batch, 16, 64] normalized streaming tokens

        Returns:
            [batch] outputs in standardized target units
        """
        m1 = None
        m2 = None

        voltage_total = 0.0

        for t in range(x.shape[1]):
            current1 = self.fc1(x[:, t])

            if m1 is None:
                m1 = torch.zeros_like(current1)

            v1 = 0.875 * m1 + current1
            s1 = (v1 >= 0.5).to(v1.dtype)
            m1 = v1 - 0.5 * s1

            current2 = self.fc2(s1)

            if m2 is None:
                m2 = torch.zeros_like(current2)

            v2 = 0.875 * m2 + current2
            s2 = (v2 >= 0.5).to(v2.dtype)
            m2 = v2 - 0.5 * s2

            voltage_total = voltage_total + v2

        output = self.out(
            voltage_total / x.shape[1]
        ).flatten()

        if self.skip is not None:
            output = output + self.skip(
                x.mean(dim=1)
            ).flatten()

        return output


def encode_rf(frame):
    """
    Encode one ORIGINAL unnormalized RF frame.

    Input:
        frame: [384, 64, 250]

    Output:
        [16, 64]

    No per-frame maximum normalization is applied.
    """
    frame = torch.as_tensor(frame).float()

    event_ids = torch.as_tensor(
        np.linspace(0, 383, 8, dtype=int),
        dtype=torch.long,
        device=frame.device,
    )

    x = frame.index_select(0, event_ids)
    x = F.pad(x, (0, 6))

    x = x.reshape(8, 8, 8, 16, 16)

    return (
        x.abs()
        .mean(dim=(2, 4))
        .permute(2, 0, 1)
        .reshape(16, 64)
    )


def load_model(folder, device="cpu"):
    """
    Load exported SNN and configuration.

    Returns:
        model, config
    """
    folder = Path(folder)

    with open(folder / "config.json") as f:
        config = json.load(f)

    checkpoint = torch.load(
        folder / "weights.pt",
        map_location=device,
        weights_only=True,
    )

    model = StreamingSNN(
        width=config["width"],
        use_skip=config["skip"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, config


@torch.no_grad()
def predict_tokens(model, tokens, config):
    """
    Predict liver-fat percentage from streaming tokens.

    tokens:
        [16,64] or [batch,16,64]

    Tokens must be derived from ORIGINAL RF amplitudes.
    """
    tokens = torch.as_tensor(tokens).float()

    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)

    assert tokens.shape[1:] == (16, 64)

    x = (
        tokens / config["gain"]
    ).clamp(0, 5)

    device = next(model.parameters()).device
    output = model(x.to(device))

    return (
        config["target_mean"] + 10 * output
    ).cpu()


@torch.no_grad()
def predict_rf(model, rf_frames, config):
    """
    Predict from raw, unnormalized RF frames.

    rf_frames:
        [384,64,250] or [N,384,64,250]

    Returns one prediction per frame.
    For rat-level predictions, average its 10
    frame predictions.
    """
    frames = torch.as_tensor(rf_frames)

    if frames.ndim == 3:
        frames = frames.unsqueeze(0)

    tokens = torch.stack([
        encode_rf(frame)
        for frame in frames
    ])

    return predict_tokens(model, tokens, config)
