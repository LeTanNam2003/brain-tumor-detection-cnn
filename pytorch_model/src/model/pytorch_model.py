import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load and pre-process image
def load_images_from_folder(folder, label, size=(224, 224)):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Augmentation + normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CNN model
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Trainning function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

    for epoch in range(epochs):
        model.train()
        start = time.time()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_train += (preds.argmax(1) == yb).sum().item()
            total_train += yb.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
                correct_val += (preds.argmax(1) == yb).sum().item()
                total_val += yb.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model_1.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Time: {time.time() - start:.2f}s")

    return train_losses, train_accuracies, val_losses, val_accuracies

# Plot the chart
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Train Acc')
    plt.plot(epochs, val_accuracies, 'ro-', label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig("result_train_512_001.png")

# Confusion matrix 
def show_confusion_matrix(model, val_loader, class_names):
    device = next(model.parameters()).device
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix_512_001.png")

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Save report to file
    with open("classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)

# ==== MAIN ====
if __name__ == "__main__":
    data_dir = "C:/Personal/final_graduate/Report/dataset/Brain_Tumor_MRI_Dataset/Training"

    normal_images, normal_labels = load_images_from_folder(os.path.join(data_dir, "normal"), label=0)
    meningioma_images, meningioma_labels = load_images_from_folder(os.path.join(data_dir, "meningioma"), label=1)
    glioma_images, glioma_labels = load_images_from_folder(os.path.join(data_dir, "glioma"), label=2)
    pituitary_images, pituitary_labels = load_images_from_folder(os.path.join(data_dir, "pituitary"), label=3)

    X = np.concatenate([normal_images, meningioma_images, glioma_images, pituitary_images])
    y = np.concatenate([normal_labels, meningioma_labels, glioma_labels, pituitary_labels])

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    x_train_tensor = torch.stack([transform(img) for img in x_train])
    x_val_tensor = torch.stack([transform(img) for img in x_val])
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=16)

    model = BrainTumorCNN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs=50
    )

    torch.save(model.state_dict(), "brain_tumor_model_1_001.pth")

    plot_results(train_losses, train_accuracies, val_losses, val_accuracies)
    show_confusion_matrix(model, val_loader, ["Normal", "Meningioma", "Glioma", "Pituitary"])
