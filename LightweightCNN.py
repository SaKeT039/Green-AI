import os
import shutil
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
def clean_codecarbon():
    try:
        temp_dir = Path(os.getenv('TEMP', "/tmp"))
        lock_file = temp_dir / ".codecarbon.lock"
        if lock_file.exists():
            lock_file.unlink()
            print(" Deleted .codecarbon.lock")

        for path in ["emissions.csv", ".codecarbon"]:
            p = Path(path)
            if p.exists():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any("codecarbon" in cmd.lower() for cmd in proc.info['cmdline']):
                    os.kill(proc.info['pid'], 9)
            except Exception:
                continue

        home_cc = Path.home() / ".codecarbon"
        if home_cc.exists():
            for file in home_cc.glob("*.lock"):
                try:
                    file.unlink()
                except Exception:
                    pass
    except Exception as e:
        print(f" CodeCarbon cleanup error: {e}")

clean_codecarbon()

# Model
class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LightCNN(num_classes=10).to(device)

transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1809, 0.1601, 0.1533])
])

dataset = EuroSAT(root="./data", download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

#CodeCarbon Tracker
if not os.path.exists("."):
    os.makedirs(".")

tracker = EmissionsTracker(
    project_name="EuroSAT-LightCNN",
    output_dir=".",
    save_to_file=True,
    log_level="error",  
    allow_multiple_runs=True
)

#Training
num_epochs = 30
patience_limit = 5
best_val_loss = float('inf')
initial_val_loss = None
patience_counter = 0
train_acc_list, val_acc_list = [], []

tracker.start()

try:
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}] Training", leave=False, dynamic_ncols=True)
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(1) == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += batch_size

        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        val_acc_list.append(val_acc)

        scheduler.step(val_loss)

        if epoch == 1:
            initial_val_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered!")
                break

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

finally:
    try:
        emissions = tracker.stop()
    except Exception as e:
        print(f"Tracker stop error: {e}")
        emissions = None


print("\n FINAL REPORT")
if initial_val_loss:
    loss_reduction = (initial_val_loss - best_val_loss) / initial_val_loss * 100
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Validation Loss Improved by: {loss_reduction:.2f}%")

if emissions is not None:
    print(f"Carbon Emissions: {emissions:.6f} kg CO₂eq")
    print(f"Energy Consumption: {emissions * 0.527:.6f} kWh (approx)")

print(f"Average Training Accuracy: {np.mean(train_acc_list):.2f}%")
print(f"Average Validation Accuracy: {np.mean(val_acc_list):.2f}%")
print("\nTraining Complete with Green AI Optimization!")


epochs = range(1, len(train_acc_list) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc_list, label='Train Accuracy')
plt.plot(epochs, val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()