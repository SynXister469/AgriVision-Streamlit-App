import os
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "leaf_disease_model.h5")
TFLITE_PATH = os.path.join(BASE_DIR, "leaf_disease_model.tflite")


model = tf.keras.models.load_model(MODEL_PATH)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # optional optimization
tflite_model = converter.convert()


with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("Conversion successful!")
print(f"TFLite model saved to: {TFLITE_PATH}")
