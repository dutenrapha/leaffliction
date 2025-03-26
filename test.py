import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input,
    Multiply,
    Flatten,
    Activation,
    Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
# import subprocess


if tf.config.list_physical_devices("GPU"):
    print("GPU is available for training.")
else:
    print("GPU is not available. Training will be done on CPU.")


def attention_layer(inputs):
    """
    Applies an attention mechanism to the input tensor.

    :param inputs: Input tensor.
    :return: Tensor after applying attention.
    """
    attention_scores = Dense(1, activation="sigmoid")(inputs)
    attention_scores = Flatten()(attention_scores)
    attention_weights = Activation("softmax")(attention_scores)
    attention_weights = Reshape((inputs.shape[1], inputs.shape[2],
                                 1))(attention_weights)
    attention_output = Multiply()([inputs, attention_weights])
    return attention_output


def train_model(directory, model_name):
    """
    Trains a model with EfficientNetB0 and custom attention.

    :param directory: Path to the directory containing image subdirectories.
    :param model_name: Name used to save the trained model.
    """
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    valid_data = train_datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    base_model = EfficientNetB0(
        input_shape=(150, 150, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Congela o backbone

    inputs = Input(shape=(150, 150, 3))
    x = base_model(inputs, training=False)
    x = attention_layer(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    model.fit(
        train_data,
        epochs=150,
        validation_data=valid_data,
        callbacks=[early_stopping]
    )

    model.save(f"model_leaves_effnet_attention_{model_name}.h5")


if __name__ == '__main__':
    train_model("./transformed_images/apple", "apple")
    train_model("./transformed_images/grape", "grape")
