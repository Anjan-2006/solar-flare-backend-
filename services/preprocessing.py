import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    REQUIRED_COLUMNS = [
        'xrsa_flux', 'xrsb_flux', 'IMF_magnitude', 'Bx', 'By', 'Bz',
        'flow_speed', 'proton_density', 'proton_temperature'
    ]

    @classmethod
    def clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Check required columns
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
                 
        # Forward fill and then backward fill missing values
        df = df.ffill().bfill()
        
        # Ensure correct types
        for col in cls.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df

    @classmethod
    def scale_features(cls, df: pd.DataFrame, feature_columns: list) -> np.ndarray:
        # For production you'd load a saved scaler here
        # E.g. scaler = joblib.load('scaler.pkl')
        # Here we just initialize one for the pipeline demonstration
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[feature_columns])
        return scaled_data
