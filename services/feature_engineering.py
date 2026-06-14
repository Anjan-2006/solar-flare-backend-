import pandas as pd
import numpy as np

class FeatureEngineer:
    FINAL_FEATURES = [
        'xrsa_flux', 'xrsb_flux', 'IMF_magnitude', 'Bx', 'By', 'Bz',
        'flow_speed', 'proton_density', 'proton_temperature',
        'log_xrsb', 'log_xrsa',
        'xrsb_lag_1', 'xrsa_lag_1', 'xrsb_lag_5', 'xrsa_lag_5', 'xrsb_lag_10', 'xrsa_lag_10',
        'xrsb_roll_mean_5', 'xrsb_roll_std_5', 'xrsb_roll_mean_10', 'xrsb_roll_std_10',
        'xrsb_roll_mean_30', 'xrsb_roll_std_30',
        'xrsb_diff', 'xrsa_diff'
    ]

    @classmethod
    def generate_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Ensure positive values before taking log
        df['log_xrsb'] = np.log1p(np.maximum(df['xrsb_flux'], 0))
        df['log_xrsa'] = np.log1p(np.maximum(df['xrsa_flux'], 0))
        
        # Lag features
        df['xrsb_lag_1'] = df['xrsb_flux'].shift(1)
        df['xrsa_lag_1'] = df['xrsa_flux'].shift(1)
        df['xrsb_lag_5'] = df['xrsb_flux'].shift(5)
        df['xrsa_lag_5'] = df['xrsa_flux'].shift(5)
        df['xrsb_lag_10'] = df['xrsb_flux'].shift(10)
        df['xrsa_lag_10'] = df['xrsa_flux'].shift(10)
        
        # Rolling averages and standard deviation
        df['xrsb_roll_mean_5'] = df['xrsb_flux'].rolling(window=5).mean()
        df['xrsb_roll_std_5'] = df['xrsb_flux'].rolling(window=5).std()
        df['xrsb_roll_mean_10'] = df['xrsb_flux'].rolling(window=10).mean()
        df['xrsb_roll_std_10'] = df['xrsb_flux'].rolling(window=10).std()
        df['xrsb_roll_mean_30'] = df['xrsb_flux'].rolling(window=30).mean()
        df['xrsb_roll_std_30'] = df['xrsb_flux'].rolling(window=30).std()
        
        # Differencing
        df['xrsb_diff'] = df['xrsb_flux'].diff()
        df['xrsa_diff'] = df['xrsa_flux'].diff()

        # Handle NaNs spawned by shifts and rolling computations
        df.bfill(inplace=True)
        df.fillna(0, inplace=True) # fallback

        # Return strictly the 25 required features in exact order
        return df[cls.FINAL_FEATURES]
