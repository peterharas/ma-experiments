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
        dense_layers,
        num_blocks=2,
        architecture="slstm_second"
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        if architecture == "slstm_first":
            num_blocks = 2
            slstm_at = [0]

        elif architecture == "slstm_second":
            num_blocks = 2
            slstm_at = [1]

        elif architecture == "only_slstm":
            num_blocks = 1
            slstm_at = [0]

        elif architecture == "only_mlstm":
            num_blocks = 1
            slstm_at = []      

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
            embedding_dim=hidden_size,
            num_blocks=num_blocks,
            slstm_at=slstm_at
        )

        self.backbone = xLSTMBlockStack(cfg)

        dense_modules = []
        for _ in range(dense_layers):
            dense_modules.append(nn.Linear(hidden_size, hidden_size))
            dense_modules.append(nn.ReLU())

        self.dense_stack = nn.Sequential(*dense_modules)

        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        # x shape:
        # [batch, seq_len, features]

        x = self.input_proj(x)

        x = self.backbone(x)

        # take last timestep
        x = x[:, -1, :]

        x = self.dense_stack(x)

        x = self.output_layer(x)

        return x