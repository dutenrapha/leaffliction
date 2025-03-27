import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)
from transformers import ViTForImageClassification


def predict_directory(dir_path, model_path, image_size=(224, 224), device=None):
    """
    Predicts images in a directory using a pre-trained PyTorch model.

    Args:
        dir_path (str): Path to the directory containing class folders.
        model_path (str): Path to the trained PyTorch model (.pth).
        image_size (tuple): Target size for image resizing (width, height).
        device (str or torch.device): Device to run inference on.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory '{dir_path}' does not exist")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model '{model_path}' does not exist")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")


    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=4
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    classes = [d for d in sorted(os.listdir(dir_path))
               if os.path.isdir(os.path.join(dir_path, d))]

    if not classes:
        raise ValueError("No class folders found in the directory")

    true_labels = []
    predicted_labels = []

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
                img = Image.open(image_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = F.softmax(outputs.logits, dim=1)
                    predicted_class_index = torch.argmax(probs, dim=1).item()

                true_labels.append(class_index)
                predicted_labels.append(predicted_class_index)

            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")

    if not true_labels:
        raise ValueError("No images were successfully processed")

    # Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
    plt.title("Confusion Matrix")

    metrics_text = (
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1-Score: {f1:.2f}"
    )
    plt.gca().text(1.2, 0.6, metrics_text, transform=plt.gca().transAxes,
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

if __name__ == '__main__':
    predict_directory('./test_dataset_apple', 'model_vit_hf_apple_final.pth')
    predict_directory('./test_dataset_grape', 'model_vit_hf_grape_final.pth')