import torch
from torch.utils.data import Dataset

class TFTCustomDatasetWeather(Dataset):
    def __init__(self, X, y, future, known_indices):
        """
        X: (Num_Samples, 168, num_features)
        y: (Num_Samples, 96)
        future: (Num_Samples, 96, 2)
        known_indices: List of integers, the column indices in X that correspond to 'future'
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.future = torch.tensor(future, dtype=torch.float32)
        
        self.window_len = self.X.shape[1]
        self.forecast_len = self.y.shape[1]
        
        # Figure out which columns are unknown (past only) and known (past + future)
        self.known_indices = known_indices
        num_total_features = self.X.shape[-1]
        self.unknown_indices = [i for i in range(num_total_features) if i not in known_indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_seq = self.X[idx]
        y_seq = self.y[idx]
        future_seq = self.future[idx] 
        
        # 1. Split the past sequence (X) into Unknown and Known features
        unknown_past = X_seq[:, self.unknown_indices]
        known_past = X_seq[:, self.known_indices]
        
        # Encoder gets the real unknown past + the real known past
        enc_cont = torch.cat([unknown_past, known_past], dim=-1)

        # 2. Prepare Decoder (Future) Features
        # Pad the unknown features with zeros for the future window
        unknown_dec_pad = torch.zeros((self.forecast_len, unknown_past.shape[-1]), dtype=torch.float32)
        
        # Decoder gets the zero-padded unknown future + the real future variables
        dec_cont = torch.cat([unknown_dec_pad, future_seq], dim=-1)

        # 3. Build the FULL dictionary
        x = {
            # Continuous Variables are now perfectly aligned!
            "encoder_cont": enc_cont, 
            "decoder_cont": dec_cont, 
            
            # Categorical Variables 
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