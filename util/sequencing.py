import numpy as np


def create_sequences(X, y, window_len, forecast_steps):
    Xs, ys = [], []
    max_horizon = max(forecast_steps)

    for i in range(len(X) - window_len - max_horizon + 1):
        X_window = X.iloc[i:i + window_len].values
        targets = np.array([y.iloc[i + window_len + h - 1] for h in forecast_steps])

        # Skip if any NaNs in inputs or targets
        if np.isnan(X_window).any() or np.isnan(targets).any():
            continue

        Xs.append(X_window)
        ys.append(targets)

    return np.array(Xs), np.array(ys)
