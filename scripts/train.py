import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset.download import download_and_extract_kvasir
from dataset.loader import load_dataset, KvasirSegDataset
from model.unet import UNet
from utils.metrics import dice_score, iou_score
from utils.visualize import plot_prediction

# ==== Configuration ====
DATASET_PATH = "kvasir-seg"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
MASK_DIR = os.path.join(DATASET_PATH, "masks")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "models/unet_kvasirseg.pth"

# ==== Download Dataset ====
download_and_extract_kvasir()

# ==== Load Data ====
images, masks = load_dataset(IMAGE_DIR, MASK_DIR)
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = KvasirSegDataset(images, masks, transform=None)

# Split dataset into train and val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# ==== Initialize Model ====
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Training Loop ====
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for inputs, masks in train_loader:
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)

    # ==== Validation ====
    model.eval()
    val_dice = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            outputs = model(inputs)
            val_dice += dice_score(outputs, masks).item()
            val_iou += iou_score(outputs, masks).item()

    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

# ==== Save Model ====
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# ==== Plot Sample Output ====
sample_inputs, sample_masks = next(iter(val_loader))
sample_inputs, sample_masks = sample_inputs.to(DEVICE), sample_masks.to(DEVICE)
sample_outputs = torch.sigmoid(model(sample_inputs))
plot_prediction(sample_inputs.cpu(), sample_masks.cpu(), sample_outputs.cpu())
