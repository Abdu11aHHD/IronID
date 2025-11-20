import os
import shutil
import random
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- Configuration ---
IMG_SIZE = (224, 224)
RANDOM_SEED = 42
AUG_PER_IMAGE = 2  # Create 2 augmented versions for every 1 original image
# Split Ratios
SPLIT_TRAIN = 0.6
SPLIT_VAL = 0.2
SPLIT_TEST = 0.2

def setup_directories(base_output_dir):
    """Creates the necessary directory structure."""
    if os.path.exists(base_output_dir):
        print(f"Warning: Output directory '{base_output_dir}' already exists. Merging/Overwriting...")
    
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        path = os.path.join(base_output_dir, subset)
        os.makedirs(path, exist_ok=True)
    return subsets

def clean_and_resize_images(raw_dir, temp_clean_dir):
    """
    Reads images from raw_dir, resizes them, renames them, 
    and saves them to a temporary directory.
    """
    print(f"\n--- Step 1: Cleaning and Resizing Images ---")
    
    # Detect classes based on folders
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    print(f"Detected Classes: {classes}")

    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    
    for cls in classes:
        src_folder = os.path.join(raw_dir, cls)
        dst_folder = os.path.join(temp_clean_dir, cls)
        os.makedirs(dst_folder, exist_ok=True)

        # Gather all image paths
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(src_folder, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(src_folder, f"*{ext.upper()}")))

        idx = 0
        print(f"Processing class: {cls} ({len(image_paths)} images)")
        
        for img_path in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                
                # Filter small images
                w, h = img.size
                if min(w, h) < 80:
                    continue
                
                # Resize
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                
                # Save with standardized name
                new_name = f"{cls}_{idx:05d}.jpg"
                img.save(os.path.join(dst_folder, new_name), "JPEG", quality=90)
                idx += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return classes

def split_dataset(temp_clean_dir, final_output_dir, classes):
    """
    Splits the cleaned images into Train, Val, and Test folders.
    """
    print(f"\n--- Step 2: Splitting Dataset ---")
    
    train_dir = os.path.join(final_output_dir, "train")
    val_dir = os.path.join(final_output_dir, "val")
    test_dir = os.path.join(final_output_dir, "test")

    for cls in classes:
        cls_src = os.path.join(temp_clean_dir, cls)
        images = sorted(glob.glob(os.path.join(cls_src, "*.jpg")))
        
        total = len(images)
        if total == 0:
            print(f"Skipping {cls} (0 images)")
            continue

        # 1. Split Train vs (Val + Test)
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(1 - SPLIT_TRAIN), 
            random_state=RANDOM_SEED, 
            shuffle=True
        )

        # 2. Split Val vs Test
        # Adjust ratio because temp_imgs is smaller than total
        val_ratio_adjusted = SPLIT_VAL / (SPLIT_VAL + SPLIT_TEST)
        
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=(1 - val_ratio_adjusted),
            random_state=RANDOM_SEED,
            shuffle=True
        )

        # Copy files to destination
        split_map = [
            (train_dir, train_imgs),
            (val_dir, val_imgs),
            (test_dir, test_imgs)
        ]

        for out_dir, img_list in split_map:
            cls_out = os.path.join(out_dir, cls)
            os.makedirs(cls_out, exist_ok=True)
            for img in img_list:
                shutil.copy(img, os.path.join(cls_out, os.path.basename(img)))

        print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

def augment_training_data(final_output_dir):
    """
    Applies data augmentation only to the training set.
    """
    print(f"\n--- Step 3: Augmenting Training Data ---")

    train_base = os.path.join(final_output_dir, "train")
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    classes = sorted([d for d in os.listdir(train_base) if os.path.isdir(os.path.join(train_base, d))])

    for cls in classes:
        cls_path = os.path.join(train_base, cls)
        img_paths = glob.glob(os.path.join(cls_path, "*.jpg"))
        
        # Filter out images that are already augmented (if running script multiple times)
        img_paths = [p for p in img_paths if "aug_" not in os.path.basename(p)]

        print(f"Augmenting {cls} ({len(img_paths)} originals)...")

        for img_path in tqdm(img_paths):
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                i = 0
                for batch in datagen.flow(
                    x,
                    batch_size=1,
                    save_to_dir=cls_path,
                    save_prefix="aug",
                    save_format="jpg"
                ):
                    i += 1
                    if i >= AUG_PER_IMAGE:
                        break
            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")

def generate_report(final_output_dir):
    """
    Prints statistics and saves a plot image.
    """
    print(f"\n--- Step 4: Generating Report ---")
    
    train_dir = os.path.join(final_output_dir, "train")
    val_dir = os.path.join(final_output_dir, "val")
    test_dir = os.path.join(final_output_dir, "test")

    classes = sorted(os.listdir(train_dir))
    rows = []

    for cls in classes:
        train_c = len(glob.glob(os.path.join(train_dir, cls, "*.jpg")))
        val_c = len(glob.glob(os.path.join(val_dir, cls, "*.jpg")))
        test_c = len(glob.glob(os.path.join(test_dir, cls, "*.jpg")))
        total = train_c + val_c + test_c
        rows.append([cls, train_c, val_c, test_c, total])

    df = pd.DataFrame(rows, columns=["Class", "Train", "Val", "Test", "Total"])
    print("\nFinal Dataset Statistics:")
    print(df.to_string(index=False))

    # Generate Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Class", y="Train", data=df, palette="viridis")
    plt.title("Number of Images per Class (TRAIN) After Augmentation")
    plt.xlabel("Class")
    plt.ylabel("Train Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(final_output_dir, "dataset_distribution.png")
    plt.savefig(plot_path)
    print(f"\nDistribution plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="IronID Data Preprocessing Pipeline")
    
    # Default paths as requested
    default_raw = "C:/Users/Abdullah/Desktop/IronID_Project/IronID/Sample_Data/raw"
    default_out = "C:/Users/Abdullah/Desktop/IronID_Project/IronID/Sample_Data/processed"

    parser.add_argument("--raw", type=str, default=default_raw, help="Path to raw image folder")
    parser.add_argument("--out", type=str, default=default_out, help="Path to processed output folder")
    
    args = parser.parse_args()

    raw_path = args.raw
    out_path = args.out
    # Create a temp folder inside the output folder to store resized, pre-split images
    temp_path = os.path.join(os.path.dirname(out_path), "temp_cleaned_data")

    print(f"Input Directory:  {raw_path}")
    print(f"Output Directory: {out_path}")
    print(f"Temp Directory:   {temp_path}")

    if not os.path.exists(raw_path):
        print(f"Error: Raw data directory not found at {raw_path}")
        return

    # 1. Setup
    setup_directories(out_path)

    # 2. Clean & Resize
    classes = clean_and_resize_images(raw_path, temp_clean_dir=temp_path)

    # 3. Split
    split_dataset(temp_path, out_path, classes)

    # 4. Augment
    augment_training_data(out_path)

    # 5. Cleanup Temp
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_path, ignore_errors=True)

    # 6. Report
    generate_report(out_path)
    print("\nPreprocessing Complete!")

if __name__ == "__main__":
    main()