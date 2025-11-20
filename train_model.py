import os
import argparse
import tf_keras as keras
import tensorflow_model_optimization as tfmot

# Import our custom modules
from src.data_loader import load_datasets
from src.models import build_mobilenet_large, apply_pruning
from src.utils import convert_to_tflite_fp16, evaluate_model

def main(args):
    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, batch_size=args.batch_size)
    
    # 2. Train Baseline
    model = build_mobilenet_large(img_shape=(224, 224, 3), num_classes=len(class_names))
    
    print("\n=== Phase 1: Training Baseline ===")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    
    # Evaluate Baseline
    evaluate_model(model, val_ds, class_names, title="Baseline_Float32")
    
    # 3. Pruning
    print("\n=== Phase 2: Pruning ===")
    pruned_model = apply_pruning(model, train_ds, args.batch_size, epochs=args.prune_epochs)
    
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Add summaries if you want TensorBoard support
    ]
    
    pruned_model.fit(train_ds, validation_data=val_ds, epochs=args.prune_epochs, callbacks=callbacks)
    
    # 4. Strip & Save Keras
    print("\n=== Phase 3: Saving Models ===")
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    keras_save_path = "models/mobilenet_pruned.keras"
    os.makedirs("models", exist_ok=True)
    final_model.save(keras_save_path)
    
    # 5. Convert to TFLite (Float16)
    tflite_save_path = "models/ironid_model_fp16.tflite"
    convert_to_tflite_fp16(keras_save_path, tflite_save_path)

    print("\n✅ Training Pipeline Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IronID Training Script")
    parser.add_argument("--data_dir", type=str, default="Sample_Data/processed", help="Path to processed data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Baseline training epochs")
    parser.add_argument("--prune_epochs", type=int, default=5, help="Fine-tuning epochs for pruning")
    
    args = parser.parse_args()
    main(args)
