import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def convert_to_tflite_fp16(keras_model_path, save_path):
    print(f"--- Converting to TFLite (Float16) ---")
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

#Float16 Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved: {save_path} ({size_mb:.2f} MB)")

def evaluate_model(model, dataset, class_names, title="Model"):
    print(f"\n--- Evaluating {title} ---")
    predictions = model.predict(dataset)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([y for x, y in dataset], axis=0)

    print(classification_report(y_true, y_pred, target_names=class_names))

#Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {title}')
    plt.savefig(f'confusionmatrix{title}.png')
    print(f"Confusion matrix saved as confusionmatrix{title}.png")