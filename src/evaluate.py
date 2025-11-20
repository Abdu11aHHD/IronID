import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model(data_dir, model_path, output_dir):
    """
    Loads test data and a trained model to generate performance metrics.
    """
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading test data from: {test_dir}")
    
    # IMPORTANT: preprocessing_function should match what you used during training!
    # Common options:
    # 1. rescal=1./255 (if you normalized pixels to 0-1)
    # 2. tf.keras.applications.resnet50.preprocess_input (if transfer learning)
    # Here we default to 1./255 as it's the most common basic scaling.
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # IMPORTANT: Do not shuffle for evaluation so labels match
    )

    # 1. Run Prediction
    print("\nRunning predictions...")
    # steps must be set to ensure we cover exactly all samples
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 2. Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)

    # Save report to text file
    with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
        f.write(report)

    # 3. Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")

def main():
    parser = argparse.ArgumentParser(description="IronID Model Evaluation Script")

    # Default paths based on your project structure
    default_data_path = os.path.join("IronID_Project", "IronID", "Sample_Data", "processed")
    # You will need to point this to your actual saved model file
    default_model_path = os.path.join("IronID_Project", "IronID", "models", "best_model.h5")
    default_output = os.path.join("IronID_Project", "IronID", "reports")

    parser.add_argument("--data", type=str, default=default_data_path, help="Path to processed data (containing 'test' folder)")
    parser.add_argument("--model", type=str, default=default_model_path, help="Path to the trained .h5 or .keras model file")
    parser.add_argument("--output", type=str, default=default_output, help="Directory to save evaluation results")

    args = parser.parse_args()

    evaluate_model(args.data, args.model, args.output)

if __name__ == "__main__":
    main()