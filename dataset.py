"""DATASET credits @SullyChen (https://github.com/SullyChen/driving-datasets)"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import imageio as iio
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

class DrivingDataset(Dataset):
    def __init__(self, labels_path, data_dir, transform=None):
        with open(Path(labels_path), "r") as f:
            self.labels = f.readlines()
            f.close()
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index) :
        img_path, label = self.labels[index].split()
        img = iio.imread(self.data_dir/img_path)

        if self.transform:
            img = self.transform(img)

        return (img, label)
        

if __name__=="__main__":
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DATA_DIR = Path("/content/data/driving_dataset_preprocessed")
        LABELS_PATH = DATA_DIR/"data.txt"
    except:
        DATA_DIR = Path("/home/avishkar/Desktop/research/driving_dataset_preprocessed")
        if not DATA_DIR.exists():
            print("Dataset doesnt exist or not preprocessed")
        else:
            LABELS_PATH = DATA_DIR/"data.txt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = DrivingDataset(labels_path=LABELS_PATH, data_dir=DATA_DIR, transform=transform)
    # print(len(dataset))
    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset)-train_size)
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    for i, (imgs, labels) in enumerate(train_loader):
        img = imgs[0]
        label = labels[0]
        plt.imshow(np.transpose(img,  (1, 2, 0)))
        plt.title(label)
        # plt.axis('off')
        plt.show()
        break
