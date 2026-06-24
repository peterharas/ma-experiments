import torch
from torch.utils.data import Dataset

class TFTCustomDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (Num_Samples, 168, num_features)
        # y shape: (Num_Samples, 96)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_len = self.X.shape[1]
        self.forecast_len = self.y.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_seq = self.X[idx]
        y_seq = self.y[idx]

        # 1. The Dummy Trick for the Decoder
        dummy_enc = torch.zeros((self.window_len, 1), dtype=torch.float32)
        dummy_dec = torch.zeros((self.forecast_len, 1), dtype=torch.float32)
        enc_cont = torch.cat([X_seq, dummy_enc], dim=-1)

        # 2. Build the FULL dictionary compliant with TimeSeriesDataSet specs
        x = {
            # Continuous Variables
            "encoder_cont": enc_cont, 
            "decoder_cont": dummy_dec, 
            
            # Categorical Variables (Empty, but strictly required by the VSNs)
            "encoder_cat": torch.empty((self.window_len, 0), dtype=torch.long),
            "decoder_cat": torch.empty((self.forecast_len, 0), dtype=torch.long),
            
            # Sequence Lengths
            "encoder_lengths": torch.tensor(self.window_len, dtype=torch.long),
            "decoder_lengths": torch.tensor(self.forecast_len, dtype=torch.long),
            
            # --- NEW ADDITIONS FOR 100% API COMPLIANCE ---
            
            # Targets (Base models sometimes check these for scaling/loss logic)
            # We just pass dummy zeros for encoder target, and real target for decoder
            "encoder_target": torch.zeros(self.window_len, dtype=torch.float32),
            "decoder_target": y_seq,
            
            # Group IDs (Required if it attempts to look up series identifiers)
            "group_ids": torch.tensor([0], dtype=torch.long),
            
            # Target Scale (Mean and Std used for un-scaling outputs)
            # Dummy values: Mean = 0.0, Std = 1.0 (Identity scaling)
            "target_scale": torch.tensor([0.0, 1.0], dtype=torch.float32)
        }
        
        # Regarding 'y': The docs specify y is a tuple (target, weight).
        # Since you are writing a custom training loop (loss = criterion(preds, batch_y)), 
        # you don't need to return the weight tuple here. Just returning the target is perfectly fine.
        return x, y_seq