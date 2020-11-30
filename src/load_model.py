import tensorflow as tf
from src.conf_loader import MODEL_DIR


def load_model(filepath=MODEL_DIR):
    return tf.keras.models.load_model(filepath, compile=False)


model = load_model()
model.summary()
