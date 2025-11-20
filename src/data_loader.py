import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

def load_datasets(data_dir, img_size=(224, 224), batch_size=32):
    print(f"--- Loading Data from: {data_dir} ---")

    train_ds = image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode='int',
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size
    )

    val_ds = image_dataset_from_directory(
        f"{data_dir}/val",
        label_mode='int',
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size
    )

    test_ds = image_dataset_from_directory(
        f"{data_dir}/test",
        label_mode='int',
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size
    )

    return train_ds, val_ds, test_ds, train_ds.class_names