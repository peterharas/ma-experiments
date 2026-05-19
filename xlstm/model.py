import torch
import torch.nn as nn

import sys

from xLSTM import xLSTMBlockStack, xLSTMBlockStackConfig
from xLSTM.blocks.mlstm.block import mLSTMBlockConfig
from xLSTM.blocks.slstm.block import sLSTMBlockConfig
from xLSTM.components.feedforward import FeedForwardConfig

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.experiment_params import *

class xLSTMForecaster(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_blocks=2
    ):
        super().__init__()

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm_dim=hidden_size,
                num_heads=4
            ),

            slstm_block=sLSTMBlockConfig(
                slstm_dim=hidden_size,
                num_heads=4
            ),

            context_length=WINDOW_LEN,
            num_blocks=num_blocks,
            embedding_dim=input_size,

            slstm_at=[1],

            feedforward=FeedForwardConfig(
                proj_factor=1.3,
                act_fn="gelu"
            )
        )

        self.backbone = xLSTMBlockStack(cfg)

        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):

        # x shape:
        # [batch, seq_len, features]

        x = self.backbone(x)

        # take last timestep
        x = x[:, -1, :]

        x = self.head(x)

        return x