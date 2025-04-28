import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class IclevrDataset(Dataset):
    def __init__(self):
        self.object_dict = json.load(open('../file/objects.json'))
        self.train_data = json.load(open('../file/train.json'))
        self.train_data = [[f'../iclevr/{key}', self.train_data[key]] for key in self.train_data.keys()]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        image = Image.open(self.train_data[idx][0]).convert("RGB")
        image = self.transform(image)
        label = torch.zeros(24)
        for obj in self.train_data[idx][1]:
            label[self.object_dict[obj]] = 1            
        return image, label
    
if __name__ == "__main__":
    dataset = IclevrDataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)