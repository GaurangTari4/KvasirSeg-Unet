import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class KvasirDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_dataloader(data_dir, batch_size=4, split='train'):
    """
    Function to get DataLoader for the Kvasir-SEG dataset.
    
    :param data_dir: Directory containing the dataset
    :param batch_size: Batch size for DataLoader
    :param split: 'train' or 'val' split
    :return: DataLoader instance for the specified dataset split
    """
    image_dir = os.path.join(data_dir, 'Kvasir-SEG', 'images')
    mask_dir = os.path.join(data_dir, 'Kvasir-SEG', 'masks')

    # Define transformations for data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = KvasirDataset(image_dir, mask_dir, transform=transform)

    # Split the dataset into training and validation sets
    total_size = len(dataset)
    val_size = int(0.2 * total_size)  # 20% for validation
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    if split == 'train':
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
