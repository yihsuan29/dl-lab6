import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class IclevrDataset(Dataset):
    def __init__(self, file_root="../file", data_root="../iclevr", mode="train"):
        self.mode = mode
        self.file_root = file_root
        self.data_root = data_root
        self.object_dict = json.load(open(os.path.join(self.file_root,'objects.json')))
        if mode == "train":
            self.data = json.load(open(os.path.join(self.file_root,'train.json')))
            # element = [dir, multi-label]
            self.data = [[os.path.join(self.data_root, key), self.data[key]] for key in self.data.keys()]
        elif mode == "test":
            self.data = json.load(open(os.path.join(self.file_root,'test.json')))        
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode=="train":
            # use element's dir to load image
            image = Image.open(self.data[idx][0]).convert("RGB")
            image = self.transform(image)
            # one-hot encoding label
            label = torch.zeros(24)
            for obj in self.data[idx][1]:
                label[self.object_dict[obj]] = 1            
            return image, label
        elif self.mode=="test":
            labels = torch.zeros(len(self.data),24)
            for i in range(len(self.data)):
                for obj in self.data[i]:
                    labels[i][self.object_dict[obj]] = 1
            return labels
        
    
    
if __name__ == "__main__":
    dataset = IclevrDataset(file_root="../file", data_root="../iclevr")
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    
    dataset = IclevrDataset(file_root="../file", data_root="../iclevr", mode="test")
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)