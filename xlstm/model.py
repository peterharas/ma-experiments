import torch
import torch.nn as nn

import sys

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.components.feedforward import FeedForwardConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.experiment_params import *

class xLSTMForecaster(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout,
        num_blocks=2
    ):
        super().__init__()

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    embedding_dim=hidden_size,
                    dropout=dropout,
                    num_heads=2
                )
            ),

            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    embedding_dim=hidden_size,
                    dropout=dropout,
                    num_heads=2
                )
            ),

            context_length=WINDOW_LEN,
            num_blocks=num_blocks,
            embedding_dim=input_size,

            slstm_at=[1]
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