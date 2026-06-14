import os
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_full_vanilla_lstm_resumed.keras")

# Force TensorFlow to only allocate memory as needed, instead of grabbing the entire VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU limitation error: {e}")

class ModelLoader:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            if os.path.exists(MODEL_PATH):
                print(f"Loading model from {MODEL_PATH}...")
                cls._model = tf.keras.models.load_model(MODEL_PATH)
                print("Model loaded successfully!")
            else:
                print(f"ERROR: Model file not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        return cls._model
