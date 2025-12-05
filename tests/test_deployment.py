import time
import numpy as np
import tensorflow as tf
import os

# Settings
MODEL_PATH = "models/ironid_model_fp16.tflite"
# Pointing to a specific image to test
TEST_IMAGE_PATH = "Sample_Data/processed/test/barbell/barbell_00000.jpg"
NUM_TEST_IMAGES = 100

def load_and_preprocess_image(path, input_shape):
    """Loads a real image and prepares it for the TFLite model."""
    print(f"🖼️ Loading image from: {path}")
    
    # 1. Read the file
    img = tf.io.read_file(path)
    # 2. Decode Jpeg
    img = tf.io.decode_jpeg(img, channels=3)
    # 3. Resize to model shape (224x224)
    img = tf.image.resize(img, [input_shape[1], input_shape[2]])
    # 4. Normalize
    img = tf.cast(img, tf.float32)
    # 5. Add Batch Dimension (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

def stress_test_tflite():
    print(f"--- 📱 Starting IronID Mobile Stress Test (Real Image) ---")
    
    # 1. Load the TFLite Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # 2. Load REAL Data
    if os.path.exists(TEST_IMAGE_PATH):
        input_data = load_and_preprocess_image(TEST_IMAGE_PATH, input_shape)
    else:
        print(f"⚠️ Warning: Image not found at {TEST_IMAGE_PATH}. Using random noise.")
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    # 3. Warmup
    print("🔥 Warming up model...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Stress Test Loop
    print(f"⚡ Running inference on {NUM_TEST_IMAGES} frames...")
    start_time = time.time()

    for i in range(NUM_TEST_IMAGES):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

    end_time = time.time()

    # 5. Calculate Metrics
    total_time = end_time - start_time
    avg_time_per_image = (total_time / NUM_TEST_IMAGES) * 1000
    fps = 1 / (avg_time_per_image / 1000)

    # 6. Show Prediction (Sanity Check)
    # We simply get the index of the highest confidence score
    predicted_index = np.argmax(output_data)
    confidence = np.max(output_data)

    print("\n--- 📊 Stress Test Results ---")
    print(f"✅ Total Time: {total_time:.2f} seconds")
    print(f"✅ Average Latency: {avg_time_per_image:.2f} ms/frame")
    print(f"✅ FPS (Est. on CPU): {fps:.2f} FPS")
    print(f"✅ Model Prediction Index: {predicted_index} (Confidence: {confidence:.2f})")
    
    if avg_time_per_image < 100:
        print("🚀 STATUS: READY FOR MOBILE (Latency < 100ms)")
    else:
        print("⚠️ STATUS: WARNING (High Latency)")

if __name__ == "__main__":
    stress_test_tflite()