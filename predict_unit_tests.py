import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k")


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean,
                             std=image_processor.image_std)
    ])


def load_model(model_path, num_classes):

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(image_path, model_path, classes):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path, num_classes=len(classes))
    transform = get_transform()

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_index = torch.argmax(probs, dim=1).item()

        print(
            f"{os.path.basename(image_path)} => "
            f"Prediction: {classes[predicted_index]}"
        )

    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")


if __name__ == '__main__':

    APPLE_CLASSES = ['Black Rot', 'Healthy', 'Cedar Apple Rust', 'Apple Scab']
    GRAPE_CLASSES = ['Black Rot', 'Esca', 'Healthy', 'Spot']

    apple_model = 'model_vit_hf_apple_final.pth'
    grape_model = 'model_vit_hf_grape_final.pth'

    predict_image('./test_images/Unit_test1/Apple_Black_rot1.JPG',
                  apple_model, APPLE_CLASSES)
    predict_image('./test_images/Unit_test1/Apple_healthy1.JPG',
                  apple_model, APPLE_CLASSES)
    predict_image('./test_images/Unit_test1/Apple_healthy2.JPG',
                  apple_model, APPLE_CLASSES)
    predict_image('./test_images/Unit_test1/Apple_rust.JPG',
                  apple_model, APPLE_CLASSES)
    predict_image('./test_images/Unit_test1/Apple_scab.JPG',
                  apple_model, APPLE_CLASSES)

    predict_image('./test_images/Unit_test2/Grape_Black_rot1.JPG',
                  grape_model, GRAPE_CLASSES)
    predict_image('./test_images/Unit_test2/Grape_Black_rot2.JPG',
                  grape_model, GRAPE_CLASSES)
    predict_image('./test_images/Unit_test2/Grape_Esca.JPG',
                  grape_model, GRAPE_CLASSES)
    predict_image('./test_images/Unit_test2/Grape_healthy.JPG',
                  grape_model, GRAPE_CLASSES)
    predict_image('./test_images/Unit_test2/Grape_spot.JPG',
                  grape_model, GRAPE_CLASSES)
