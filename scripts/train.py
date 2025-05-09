import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.loader import KvasirDataset
from model.unet import UNet
from utils.metrics import dice_coefficient
from utils.visualize import plot_prediction
from dataset.download import download_and_extract

# Download dataset
download_and_extract()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = KvasirDataset("data/Kvasir-SEG/images", "data/Kvasir-SEG/masks", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(1):
    model.train()
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    sample_imgs, sample_masks = next(iter(loader))
    sample_imgs, sample_masks = sample_imgs.to(device), sample_masks.to(device)
    preds = model(sample_imgs)
    plot_prediction(sample_imgs[:5], sample_masks[:5], preds[:5])
