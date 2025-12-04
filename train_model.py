import os
import argparse
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from tf_keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from src.data_loader import load_datasets
from src.models import build_mobilenet_large, apply_pruning
from src.utils import convert_to_tflite_fp16, evaluate_model, setup_logger 

def main(args):
    # 1. Setup Logging
    logger = setup_logger("training_pipeline") # START LOGGER
    logger.info("🚀 IronID Training Started")

    # 2. Load Data
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, batch_size=args.batch_size)
    logger.info(f"Data loaded. Classes: {class_names}")

    # 3. Train Baseline
    model = build_mobilenet_large(img_shape=(224, 224, 3), num_classes=len(class_names))
    
    logger.info("=== Phase 1: Training Baseline ===")

    # Define hooks
    baseline_callbacks = [
        CSVLogger('logs/baseline_training_log.csv', append=True),
        ModelCheckpoint('models/best_baseline.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    # Add callbacks
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=baseline_callbacks)
    
    # Evaluate Baseline
    evaluate_model(model, val_ds, class_names, title="Baseline_Float32")
    logger.info("Baseline evaluation complete.")
    
    # 4. Pruning
    logger.info("=== Phase 2: Pruning ===")
    pruned_model = apply_pruning(model, train_ds, args.batch_size, epochs=args.prune_epochs)
    
    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        CSVLogger('logs/pruning_log.csv', append=True) # <--- Log pruning progress too
    ]
    
    pruned_model.fit(train_ds, validation_data=val_ds, epochs=args.prune_epochs, callbacks=pruning_callbacks)
    
    # 5. Strip & Save Keras
    logger.info("=== Phase 3: Saving Models ===")
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    keras_save_path = "models/mobilenet_pruned.keras"
    os.makedirs("models", exist_ok=True)
    final_model.save(keras_save_path)
    
    # 6. Convert to TFLite (Float16)
    tflite_save_path = "models/ironid_model_fp16.tflite"
    convert_to_tflite_fp16(keras_save_path, tflite_save_path)

    logger.info("✅ Training Pipeline Complete!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="IronID Training Script")
    parser.add_argument("--data_dir", type=str, default="Sample_Data/processed", help="Path to processed data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Baseline training epochs")
    parser.add_argument("--prune_epochs", type=int, default=5, help="Fine-tuning epochs for pruning")
    
    args = parser.parse_args()
    main(args)