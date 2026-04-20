WINDOW_LEN = 7 * 24 * 4  # Window length (hours), to be tuned? *4 because of 15 min interval

TARGET_COL = 'discharge'

FORECAST_DAYS = [1, 2, 3, 4]
FORECAST_15MS = [d * 24 * 4 for d in FORECAST_DAYS]