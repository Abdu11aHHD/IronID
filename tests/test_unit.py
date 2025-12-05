import unittest
import os
import tensorflow as tf
from src.data_loader import load_datasets

class TestIronIDSystem(unittest.TestCase):

    def test_folder_structure(self):
        """Test 1: Check if essential folders exist"""
        required_folders = ["models", "src", "logs", "Sample_Data"]
        for folder in required_folders:
            self.assertTrue(os.path.exists(folder), f"Missing folder: {folder}")

    def test_model_artifact_exists(self):
        """Test 2: Check if the TFLite model was actually generated"""
        model_path = "models/ironid_model_fp16.tflite"
        self.assertTrue(os.path.exists(model_path), "TFLite model not found! Pipeline failed.")

    def test_data_loader_integrity(self):
        """Test 3: Verify Data Loader returns correct shape"""
        # We assume processed data exists from your previous runs
        data_dir = "Sample_Data/processed"
        if os.path.exists(data_dir):
            train_ds, _, _, class_names = load_datasets(data_dir, batch_size=2)
            
            # Check we have 10 classes
            self.assertEqual(len(class_names), 10, f"Expected 10 classes, found {len(class_names)}")
            
            # Check image shape (Batch, 224, 224, 3)
            for images, labels in train_ds.take(1):
                self.assertEqual(images.shape, (2, 224, 224, 3))
                break

if __name__ == '__main__':
    unittest.main()