import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root="./Data", transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"Размер тренировочной выборки: {len(train_dataset)}")
print(f"Размер тестовой выборки: {len(test_dataset)}")

test_loader_for_viz = DataLoader(test_dataset, batch_size=10, shuffle=True)

def show_images(images, labels, class_names):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(10):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

data_iter = iter(test_loader_for_viz)
images, labels = next(data_iter)

class_names = full_dataset.classes

show_images(images, labels, class_names)

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import glob

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

print("Обучение завершено.")

external_image_paths = glob.glob("./testing-images/*")
external_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

import torch.nn.functional as F

def predict_external_images(model, image_paths, transform, class_names, device):
    model.eval()
    _, axes = plt.subplots(10, 5, figsize=(12, 24))
    axes = axes.ravel()
    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                pred_label = class_names[predicted.item()]
                conf_value = confidence.item() * 100

            img_display = np.clip(input_tensor.cpu().squeeze().permute(1, 2, 0).numpy(), 0, 1)
            axes[i].imshow(img_display)
            axes[i].set_title(f"Предсказание: {pred_label}")
            axes[i].axis('off')
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if external_image_paths:
    predict_external_images(model, external_image_paths, external_transform, class_names, device)
