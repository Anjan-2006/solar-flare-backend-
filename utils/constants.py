FEATURES = [
    'xrsa_flux', 'xrsb_flux', 'IMF_magnitude', 'Bx', 'By', 'Bz', 
    'flow_speed', 'proton_density', 'proton_temperature', 'log_xrsb', 
    'log_xrsa', 'xrsb_lag_1', 'xrsa_lag_1', 'xrsb_lag_5', 'xrsa_lag_5', 
    'xrsb_lag_10', 'xrsa_lag_10', 'xrsb_roll_mean_5', 'xrsb_roll_std_5', 
    'xrsb_roll_mean_10', 'xrsb_roll_std_10', 'xrsb_roll_mean_30', 
    'xrsb_roll_std_30', 'xrsb_diff', 'xrsa_diff'
]

SEQUENCE_LENGTH = 120
NUM_FEATURES = 25
THRESHOLD = 0.48
MODEL_PATH = "models/final_full_vanilla_lstm_resumed.keras"
