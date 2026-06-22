from data_utils import InputTypes, DataTypes, FeatureSpec


# TODO: write custom config

class KarstSpringConfig():
    def __init__(self):
    
    # TODO: set features
        self.features = [
            # Time index
            FeatureSpec('timestamp', InputTypes.TIME, DataTypes.CONTINUOUS),
            
            # Target variable (Assumed to be conductivity, adjust if necessary)
            FeatureSpec('conductivity', InputTypes.TARGET, DataTypes.CONTINUOUS),
            
            # Observed dynamic covariates (unknown in the future)
            FeatureSpec('discharge', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('temperature', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('sh', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('rr', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('tl', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('delta_sh', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            
            # Known dynamic covariates (fully predictable cyclical features)
            FeatureSpec('day_sin', InputTypes.KNOWN, DataTypes.CONTINUOUS),
            FeatureSpec('day_cos', InputTypes.KNOWN, DataTypes.CONTINUOUS),
            FeatureSpec('mth_sin', InputTypes.KNOWN, DataTypes.CONTINUOUS),
            FeatureSpec('mth_cos', InputTypes.KNOWN, DataTypes.CONTINUOUS),
            
            # Lagged features (historical observations)
            FeatureSpec('rr_lagged', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('tl_lagged', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('sh_lagged', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
        ]
        
        self.example_length = 8 * 24  # Total sequence = 192 hours (8 days)
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.1
        self.attn_dropout = 0.0

        self.static_categorical_inp_lens = []
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])