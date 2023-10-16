import numpy as np 
import pandas as pd
import os

import PIL
import PIL.ImageOps
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

import copy

import torch
import torchvision.datasets as dset
from torchvision import transforms
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import random_split, SubsetRandomSampler, DataLoader
from torchsummary import summary

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device 

random_seed = 42
np.random.seed(random_seed)

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

IMG_H = 200
IMG_W = 200

dataset = dset.ImageFolder(
    root='/kaggle/input/covidct',
    transform=transforms.Compose([
               transforms.Resize((IMG_H, IMG_W)), # Resize image
               transforms.ToTensor(), # Converts images to PyTorch tensors
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize the image by subtracting the mean and dividing by the standard deviation
    ])
)

# Splitting the dataset for training and testing
train_dataset, test_dataset = random_split(
    dataset=dataset, 
    lengths = [600, 146],
    generator=torch.Generator().manual_seed(random_seed)
)

# Helper class to invert an image
def invert_img(img):
    img = PIL.ImageOps.invert(img)
    return img

class Invert(object):
    def __init__(self):
        pass
    
    def __call__(self, img):
        return invert_img(img)


train_transformation = transforms.Compose([
    transforms.RandomRotation(5), # Applies a random rotation to the image
    Invert(), # Inverts the colors of the image
    transforms.RandomAffine( # Applies a random affine transformation to the image
        degrees=0,
        scale=(1.1, 1.1), 
        shear=0.9
    ),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Apply transformations to copy of dataset
augments = copy.deepcopy(train_dataset) 
augments.dataset.transform = train_transformation 

# Concatenate both datasets 
final_train_dataset =  torch.utils.data.ConcatDataset([train_dataset,augments])