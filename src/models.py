import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models
import tensorflow_model_optimization as tfmot

def build_mobilenet_large(img_shape, num_classes):
    print("--- Building MobileNetV3-Large ---")
    preprocess = keras.applications.mobilenet_v3.preprocess_input
    base = keras.applications.MobileNetV3Large(input_shape=img_shape, include_top=False, weights='imagenet')
    base.trainable = False 

    inputs = layers.Input(shape=img_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="MobileNetV3Large")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def apply_pruning(model, train_dataset, batch_size, epochs=10):
    print("--- Applying Pruning ---")
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    num_images = tf.data.experimental.cardinality(train_dataset).numpy() * batch_size
    end_step = int(num_images / batch_size) * epochs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.00,
            final_sparsity=0.50,
            begin_step=0,
            end_step=end_step
        )
    }

    def prune_layer(layer):
        if isinstance(layer, (layers.Dense, layers.Conv2D)):
            return prune_low_magnitude(layer, **pruning_params)
        return layer

    model_for_pruning = models.clone_model(model, clone_function=prune_layer)

    model_for_pruning.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model_for_pruning