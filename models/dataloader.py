import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

import torch.nn.functional as nnf
import torchvision.transforms.functional as TF

class Shadow_Removal(Dataset):
    def __init__(self, train_data):
        self.total = unpickle('/content/drive/My Drive/Colab Notebooks/Shadow_Removal/data.pickle')
        self.transform = transforms.Compose([transforms.Normalize((127, 127, 127), (128, 128, 128))])
        if train_data == True:
            self.image = torch.from_numpy(self.total[1][0]).view(int(self.total[1][0].shape[0]/3), 3 , self.total[1][0].shape[1], self.total[1][0].shape[2])
            self.gray = torch.from_numpy(np.where(self.total[1][1] > 0, 1, self.total[1][1])).view(int(self.total[1][1].shape[0]/480), 480 , self.total[1][1].shape[1]) #Gray2Binary
            self.ground_truth = torch.from_numpy(self.total[1][2]).view(int(self.total[1][2].shape[0]/3), 3 , self.total[1][2].shape[1], self.total[1][2].shape[2])
        else:
            self.image = torch.from_numpy(self.total[0][0]).view(int(self.total[0][0].shape[0]/3), 3 , self.total[0][0].shape[1], self.total[0][0].shape[2])
            self.gray = torch.from_numpy(np.where(self.total[0][1] > 0, 1, self.total[0][1])).view(int(self.total[0][1].shape[0]/480), 480 , self.total[0][1].shape[1])
            self.ground_truth = torch.from_numpy(self.total[0][2]).view(int(self.total[0][2].shape[0]/3), 3 , self.total[0][2].shape[1], self.total[0][2].shape[2])
        self.length = self.image.shape[0]
        del self.total

    def random_crop(self, img , gray):
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(100, 120))
        img = TF.crop(img, i, j, h, w)
        gray = TF.crop(gray, i, j, h, w)
        return img,gray

    def resizing(self, img): #resize img to half
        img_PIL = transforms.ToPILImage(mode = "RGB")(img)
        img_PIL = torchvision.transforms.Resize([int(img.size()[1]/4),int(img.size()[2]/4)])(img_PIL)
        img_PIL = np.array(img_PIL)
        img_PIL = np.transpose(img_PIL, (2,0,1))
        return torch.from_numpy(img_PIL).float()

    def gray_resizing(self, img): #resize gray to half
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = torchvision.transforms.Resize([int(img.size()[0]/4),int(img.size()[1]/4)])(img_PIL)
        img_PIL = np.array(img_PIL)
        return torch.from_numpy(img_PIL).view(1, img_PIL.shape[0], img_PIL.shape[1]).float()

    def __getitem__(self, index):
        img = torch.cat((self.transform(self.resizing(self.image[index])),self.gray_resizing(self.gray[index])), 0)
        return img, self.transform(self.resizing(self.ground_truth[index]))

    def __len__(self):
        return self.length
