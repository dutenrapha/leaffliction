import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)


def predict_directory(dir_path, model_path):
    """
    Predicts images in a directory using a pre-trained model.

    Args:
        dir_path (str): Path to the directory containing class folders.
        model_path (str): Path to the trained model file.
    """
    # Verify if the directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory '{dir_path}' does not exist")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model '{model_path}' does not exist")

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Define class labels based on folder names
    classes = [d for d in sorted(os.listdir(dir_path))
               if os.path.isdir(os.path.join(dir_path, d))]

    if not classes:
        raise ValueError("No class folders found in the directory")

    true_labels = []
    predicted_labels = []

    # Iterate through class folders and images
    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(dir_path, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            print(f"Warning: No images found in the folder '{class_name}'")
            continue

        for image_name in images:
            image_path = os.path.join(class_path, image_name)
            try:
                # Preprocess the image
                img = tf.keras.preprocessing.image.load_img(
                    image_path, target_size=(150, 150)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Make prediction
                prediction = model.predict(img_array, verbose=0)
                predicted_class_index = np.argmax(prediction)

                # Store true and predicted labels
                true_labels.append(class_index)
                predicted_labels.append(predicted_class_index)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")

    if not true_labels:
        raise ValueError("No images were successfully processed")

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels,
                                average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Generate and plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
    plt.title("Confusion Matrix")

    # Add performance metrics to the plot
    metrics_text = (
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1-Score: {f1:.2f}"
    )
    plt.gca().text(1.2, 0.6, metrics_text, transform=plt.gca().transAxes,
                   verticalalignment='top', bbox=dict(facecolor='white',
                                                      alpha=0.5))

    plt.show()


def predict_image(image_path, model_path, classes):
    """
    Predict a single image using a pre-trained model.

    Args:
        image_path (str): Path to the image.
        model_path (str): Path to the trained model.
        classes (list): List of class labels.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(150, 150)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    print(f"Prediction: {classes[np.argmax(prediction)]}")


# Unit tests
APPLE_CLASSES = ['Black Rot', 'Healthy', 'Cedar Apple Rust', 'Apple Scab']
GRAPE_CLASSES = ['Black Rot', 'Esca', 'Healthy', 'Spot']

apple_model = 'model_leaves_attention_apple.h5'
grape_model = 'model_leaves_attention_grape.h5'


# Apple unit tests
predict_image('./test_images/Unit_test1/Apple_Black_rot1.JPG', apple_model,
              APPLE_CLASSES)
predict_image('./test_images/Unit_test1/Apple_healthy1.JPG', apple_model,
              APPLE_CLASSES)
predict_image('./test_images/Unit_test1/Apple_healthy2.JPG', apple_model,
              APPLE_CLASSES)
predict_image('./test_images/Unit_test1/Apple_rust.JPG', apple_model,
              APPLE_CLASSES)
predict_image('./test_images/Unit_test1/Apple_scab.JPG', apple_model,
              APPLE_CLASSES)

# Grape unit tests
predict_image('./test_images/Unit_test2/Grape_Black_rot1.JPG', grape_model,
              GRAPE_CLASSES)
predict_image('./test_images/Unit_test2/Grape_Black_rot2.JPG', grape_model,
              GRAPE_CLASSES)
predict_image('./test_images/Unit_test2/Grape_Esca.JPG', grape_model,
              GRAPE_CLASSES)
predict_image('./test_images/Unit_test2/Grape_healthy.JPG',
              grape_model, GRAPE_CLASSES)
predict_image('./test_images/Unit_test2/Grape_spot.JPG',
              grape_model, GRAPE_CLASSES)
