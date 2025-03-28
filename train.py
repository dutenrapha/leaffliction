import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTFeatureExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ GPU available" if torch.cuda.is_available() else "⚠️ Training on CPU")


def train_model(data_dir, model_name, num_epochs=50,
                batch_size=32, lr=1e-4, patience=5):

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean,
                             std=feature_extractor.image_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean,
                             std=feature_extractor.image_std)
    ])

    full_dataset = datasets.ImageFolder(root=data_dir,
                                        transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset,
                                              [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=4)

    for param in model.vit.parameters():
        param.requires_grad = False
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss /= total_val
        val_accuracy = correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss:"
              f"{train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}"
                      f" with Val Loss: {val_loss:.4f}")
                break

    torch.save(model.state_dict(), f"model_vit_hf_{model_name}_final.pth")
    print("Training completed.")


if __name__ == '__main__':
    import subprocess
    import time

    inicio = time.time()

    subprocess.run(['python', 'Augmentation.py', 'images'], check=True)

    train_model("./augmented_images/apple", "apple",
                num_epochs=30, batch_size=32, lr=1e-4, patience=5)
    train_model("./augmented_images/grape", "grape",
                num_epochs=30, batch_size=32, lr=1e-4, patience=5)

    fim = time.time()
    print(f"Training completed in {fim - inicio} seconds")
