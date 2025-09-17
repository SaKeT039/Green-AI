import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from codecarbon import EmissionsTracker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NAS_EarlyExitNet().to(device)

# Data Augmentation & Normalization 
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Updated Dataloaders with Transform
train_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  

# CodeCarbon Tracker
tracker = EmissionsTracker(
    project_name="GreenAI-EuroSAT",
    measure_power_secs=1,
    output_file="codecarbon_log.csv",
    tracking_mode="process",
    log_level="error",
    allow_multiple_runs=True
)
tracker.start()

# Training Param
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_acc = 0
patience = 3
trigger_times = 0
num_epochs = 20

os.makedirs("checkpoints", exist_ok=True)

# Training Loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        preds, _ = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds, _ = model(x)
            loss = criterion(preds, y)

            val_loss += loss.item() * x.size(0)
            _, predicted = torch.max(preds, 1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)

    val_loss_avg = val_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss_avg)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - start_time
    time_per_step = (epoch_time / len(train_loader)) * 1000

    print(f"{len(train_loader)}/{len(train_loader)} ━━ {int(epoch_time)}s {int(time_per_step)}ms/step - "
          f"acc: {train_acc:.4f} - loss: {train_loss:.4f} - "
          f"val_acc: {val_acc:.4f} - val_loss: {val_loss_avg:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(" Early stopping triggered.")
            break

tracker.stop()
emissions_data = tracker.final_emissions_data
emissions_kg = emissions_data.emissions
energy_consumed_kwh = emissions_data.energy_consumed


print(f"\n Final Epoch: {epoch}")
print(f" Final Train Accuracy: {train_accuracies[-1]:.4f}")
print(f" Final Val Accuracy: {val_accuracies[-1]:.4f}")
print(f" CO₂ Emitted: {emissions_kg:.4f} kg")
print(f" Energy Consumed: {energy_consumed_kwh:.4f} kWh")


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()