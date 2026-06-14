import numpy as np

class SequenceGenerator:
    TIMESTEPS = 120

    @classmethod
    def generate(cls, scaled_data: np.ndarray) -> np.ndarray:
        samples = len(scaled_data)
        if samples < cls.TIMESTEPS:
            raise ValueError(f"Insufficient rows. Need at least {cls.TIMESTEPS} rows to form one sequence. Got {samples}.")
            
        X = []
        for i in range(len(scaled_data) - cls.TIMESTEPS + 1):
            X.append(scaled_data[i:(i + cls.TIMESTEPS)])
            
        return np.array(X, dtype=np.float32)
