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
        
        # Encoder gets the real features + the 1 dummy feature (Total: 15 features)
        enc_cont = torch.cat([X_seq, dummy_enc], dim=-1)

        # --- THE FIX ---
        # Decoder must ALSO have 15 features. We pad the 14 unknown features with zeros,
        # and append the 1 dummy feature at the exact same index as the encoder.
        num_unknown_features = X_seq.shape[-1]
        unknown_dec_pad = torch.zeros((self.forecast_len, num_unknown_features), dtype=torch.float32)
        dec_cont = torch.cat([unknown_dec_pad, dummy_dec], dim=-1)

        # 2. Build the FULL dictionary compliant with TimeSeriesDataSet specs
        x = {
            # Continuous Variables
            "encoder_cont": enc_cont, 
            "decoder_cont": dec_cont, # <--- Use the padded dec_cont here!
            
            # Categorical Variables (Empty, but strictly required by the VSNs)
            "encoder_cat": torch.empty((self.window_len, 0), dtype=torch.long),
            "decoder_cat": torch.empty((self.forecast_len, 0), dtype=torch.long),
            
            # Sequence Lengths
            "encoder_lengths": torch.tensor(self.window_len, dtype=torch.long),
            "decoder_lengths": torch.tensor(self.forecast_len, dtype=torch.long),
            
            # Targets 
            "encoder_target": torch.zeros(self.window_len, dtype=torch.float32),
            "decoder_target": y_seq,
            
            # Group IDs 
            "group_ids": torch.tensor([0], dtype=torch.long),
            
            # Target Scale 
            "target_scale": torch.tensor([0.0, 1.0], dtype=torch.float32)
        }
        
        return x, y_seq