import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        path_img = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path_img)
        label = self.label[index]
        # if self.mode == 'train':
            # img = transforms.RandomRotation(90)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            # img = transforms.RandomVerticalFlip()(img)
        img = transforms.ToTensor()(img).to(torch.device("cuda"))
        # print(path_img)
        return img, label
