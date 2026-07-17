import torch
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary

# Import your custom dataset (adjust the path if needed)
from tft.tft_custom_dataset_weather import TFTCustomDatasetWeather

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import TorchNormalizer

# ---------------------------------------------------------
# 1. DUMMY DATA SETUP
# ---------------------------------------------------------
BATCH_SIZE = 2
WINDOW_LEN = 168
FORECAST_LENGTH = 96

future_known_input_cols = ["rr", "tl", "sh", "delta_sh"]
unknown_input_cols = ["feature1", "feature2", "feature3", "feature4"]
input_cols = unknown_input_cols + future_known_input_cols

num_features = len(input_cols)
known_indices = [input_cols.index(c) for c in future_known_input_cols]

# Create random numpy arrays mimicking your sequencing output
X_dummy = np.random.rand(BATCH_SIZE, WINDOW_LEN, num_features)
y_dummy = np.random.rand(BATCH_SIZE, FORECAST_LENGTH)
future_dummy = np.random.rand(BATCH_SIZE, FORECAST_LENGTH, len(future_known_input_cols))

# Instantiate dataset and grab one batch
dataset = TFTCustomDatasetWeather(X_dummy, y_dummy, future_dummy, known_indices)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
example_batch, _ = next(iter(dataloader))

# ---------------------------------------------------------
# 2. MODEL SETUP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move batch dictionary to the correct device
example_batch = {k: v.to(device) for k, v in example_batch.items()}

print("Instantiating TFT Model...")
model = TemporalFusionTransformer(
    hidden_size=64,
    lstm_layers=2,
    hidden_continuous_size=64,
    attention_head_size=4,
    output_size=1, 
    loss=MAE(),
    output_transformer=TorchNormalizer(),
    max_encoder_length=WINDOW_LEN,
    x_reals=unknown_input_cols + future_known_input_cols,
    x_categoricals=[],
    time_varying_reals_encoder=unknown_input_cols + future_known_input_cols,
    time_varying_categoricals_encoder=[],
    time_varying_reals_decoder=future_known_input_cols,
    time_varying_categoricals_decoder=[],
    static_reals=[],
    static_categoricals=[],   
).to(device)

# ---------------------------------------------------------
# 3. PRINT SUMMARY
# ---------------------------------------------------------
print("\n" + "="*60)
print("TEMPORAL FUSION TRANSFORMER SUMMARY")
print("="*60)

# Pass the example batch into the summary function
summary(model, input_data=[example_batch], depth=4, col_names=("input_size", "output_size", "num_params", "trainable"))