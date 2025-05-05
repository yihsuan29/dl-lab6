import torch
import torch.nn as nn
import torchvision.models as models
from diffusion_model import DiffusionModel
import json
import os
import numpy as np
from PIL import Image

'''===============================================================
1. Title:     

DLP Spring 2025 Lab6 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You may need to modify the checkpoint's path at line 40.
You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]
Images should be normalized with:
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

==============================================================='''


class evaluation_model():
    def __init__(self, new = False, seed=30):
        # Define path
        self.epochs = 269
        self.ddpm_path = f"iclevr_checkpoint_{self.epochs}.pth"
        self.new = new
        if self.new:
            self.test_path = "../file/new_test.json"
        else:
            self.test_path = "../file/test.json"
        self.object_path = "../file/objects.json"
        
        self.report_output_path = "../results"
        self.output_path = "../images"
        os.makedirs(self.report_output_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        if self.new:
            os.makedirs(f'{self.report_output_path}/{self.epochs}epoch_new/', exist_ok=True)
            os.makedirs(f'{self.output_path}/new_test', exist_ok=True)
        else:
            os.makedirs(f'{self.report_output_path}/{self.epochs}epoch_test/', exist_ok=True)
            os.makedirs(f'{self.output_path}/test', exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        
        #modify the path to your own path
        checkpoint = torch.load('../file/checkpoint.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
        
        # Load the dataset
        self.object_dict = json.load(open(self.object_path))
        self.test_data = json.load(open(self.test_path))
        labels = torch.zeros(len(self.test_data)+1,self.classnum)
        for i in range(len(self.test_data)):
            for obj in self.test_data[i]:
                labels[i][self.object_dict[obj]] = 1
        labels = labels.to(self.device)
                
        # Load model
        self.ddpm = DiffusionModel(device=None, dataset_name='iclevr', checkpoint_name=self.ddpm_path)
        self.ddpm.nn_model.eval()
        
        # Generate images & plots
        images, intermediate_samples = self.ddpm.generate(n_samples=32, n_images_per_row=8, timesteps=500, beta1=0.0001, beta2=0.02, epoch= 'test', new=self.new)
        if self.new:
            print(f"New test: {self.eval(images.cuda(), labels.cuda())}")
        else:
            print(f"Test: {self.eval(images.cuda(), labels.cuda())}")       
        
        
        # Generate denoising images
        denoise_plots=[]
        for i in range(len(intermediate_samples)):
            denoise_plots.append(intermediate_samples[i][-1])
        denoise_plots.append(images[-1])
        
    
        self.generate_denoise(denoise_plots, labels[:,-1],new=self.new)
        self.generate_gridplot(images, labels,new=self.new)
        self.generate_plot(images, labels,new=self.new)
        
    def generate_denoise(self, images, labels, new = False):
        image_grid = np.zeros((64, 64*len(images), 3))
        for i in range(len(images)):
            image_grid[:, i*64:(i+1)*64] = images[i].detach().cpu().numpy().transpose(1,2,0)
        
        image_grid = (image_grid + 1) / 2
        image_grid = (image_grid * 255).astype(np.uint8)
        image_grid = Image.fromarray(image_grid)
        
        if new:
            image_grid.save(f'{self.report_output_path}/{self.epochs}epoch_new/denoise.png')
        else:
            image_grid.save(f'{self.report_output_path}/{self.epochs}epoch_test/denoise.png')
            
    
    def generate_gridplot(self, images, labels, new = False):
        
        image_grid = np.zeros((4*64, 8*64, 3))
        for i in range(4):
            for j in range(8):
                image_grid[i*64:(i+1)*64, j*64:(j+1)*64] = images[i*8+j].detach().cpu().numpy().transpose(1,2,0)
        
        image_grid = (image_grid + 1) / 2
        image_grid = (image_grid * 255).astype(np.uint8)
        image_grid = Image.fromarray(image_grid)
        
        if new: 
            image_grid.save(f'{self.report_output_path}/{self.epochs}epoch_new/images.png')
        else:
            image_grid.save(f'{self.report_output_path}/{self.epochs}epoch_test/images.png')
            

    def generate_plot(self, images, labels, new = False):     
        for i in range(len(images)):
            image = images[i].detach().cpu().numpy().transpose(1,2,0)
            image = (image + 1) / 2
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            if new:
                image.save(f'{self.output_path}/new_test/{i}.png')
            else:
                image.save(f'{self.output_path}/test/{i}.png')  
        
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc

if __name__ == '__main__':
    evaluation_model(seed = 30, new=False)
    evaluation_model(seed = 30, new=True) 