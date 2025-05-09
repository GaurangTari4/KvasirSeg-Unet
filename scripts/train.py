import torch
import os
from dataset.loader import get_dataloader
from model.unet import UNet
from utils.metrics import dice_coefficient, iou_score
from utils.visualize import plot_prediction
from dataset.download import download_and_extract

download_and_extract()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "models/unet_kvasirseg.pth"

train_loader = get_dataloader("data", batch_size=BATCH_SIZE, split='train')
val_loader = get_dataloader("data", batch_size=1, split='val')

model = UNet(in_channels=3, out_channels=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

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

    model.eval()
    val_dice = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            outputs = model(inputs)
            val_dice += dice_coefficient(outputs, masks).item()
            val_iou += iou_score(outputs, masks).item()

    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

sample_inputs, sample_masks = next(iter(val_loader))
sample_inputs, sample_masks = sample_inputs.to(DEVICE), sample_masks.to(DEVICE)
sample_outputs = model(sample_inputs)
plot_prediction(sample_inputs.cpu(), sample_masks.cpu(), sample_outputs.cpu())
