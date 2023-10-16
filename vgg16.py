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

import torchvision.models as models

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

def getMetrics(conf_matrix):
    tp = conf_matrix[1,1]
    fp = conf_matrix[0,1]
    tn = conf_matrix[0,0]
    fn = conf_matrix[1,0]
    print("tp:{} | fp:{} | tn:{} | fn:{}".format(tp,fp,tn,fn))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    f1 = 2*precision*recall/(precision + recall)
    return (precision, recall, f1, accuracy)

# model, images_train, images_test, device, lr=0.0001, epochs=30, batch_size=32, l2=0.00001, gamma=0.5, patience=7, folds=5

def train_model(params):
    model  = params["model"]
    images_train  = params["train"]
    images_test   = params["test"]
    device = params["device"] 
    lr     = params["lr"]
    batch_size  = params["batch_size"]
    epochs = params["epochs"]
    gamma  = params["gamma"]
    patience = params["patience"]
    folds = params["folds"]
    l2 = params["l2"]
    
    splits=KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )
    
    foldperf={
            'train_loss': [], 
            'train_acc': [], 
            'train_precision': [], 
            'train_recall': [],
            'train_f1': [], 
            'val_loss': [], 
            'val_acc': [], 
            'val_precision': [], 
            'val_recall': [], 
            'val_f1': []
    }
    
    model= nn.DataParallel(model)
    model = model.to(device)
    test_dataset = DataLoader(images_test, batch_size=batch_size, shuffle=False, num_workers=2)
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(images_train)))):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_dataset = DataLoader(images_train, batch_size=batch_size, sampler=train_sampler)
        val_dataset = DataLoader(images_train, batch_size=batch_size, sampler=val_sampler)
        
        
        train_loader = train_dataset
        val_loader = val_dataset
        test_loader = test_dataset

        nb_classes = 2
        
        history = {
            'train_loss': [], 
            'train_acc': [], 
            'train_precision': [], 
            'train_recall': [],
            'train_f1': [], 
            'val_loss': [], 
            'val_acc': [], 
            'val_precision': [], 
            'val_recall': [], 
            'val_f1': []
        }
        
        criterion = nn.BCEWithLogitsLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)  
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)

        
        print("++++++++ Training ++++++++")
        for epoch in range(epochs):
            model.train()  

            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            confusion_matrix_train = torch.zeros(nb_classes, nb_classes)
            
            #  For each batch 
            for i, (images, labels) in enumerate(train_loader):
                print("=",end="")
                
                labels = labels.float()
                images = images.to(device)  
                labels = labels.to(device)  
                
                outputs = model(images).view(-1)  
                pred = torch.sigmoid(outputs)
                pred = torch.round(pred)

                cur_train_loss = criterion(outputs, labels)  
                cur_train_acc = (pred == labels).sum().item() / batch_size

                for t, p in zip(labels.view(-1), pred.view(-1)):
                    confusion_matrix_train[t.long(), p.long()] += 1

             
                cur_train_loss.backward()  
                optimizer.step()           
                optimizer.zero_grad()      

      
                train_loss += cur_train_loss         
            print(">")
        
            precision_train, recall_train, f1_train, accuracy_train = getMetrics(confusion_matrix_train)

            
            model.eval()  
            confusion_matrix_val = torch.zeros(nb_classes, nb_classes)
            with torch.no_grad():  
                for images, labels in val_loader:
                    
                    labels = labels.float()
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images).view(-1)

                    # loss
                    cur_valid_loss = criterion(outputs, labels)
                    print("current lost :",cur_valid_loss )
                    val_loss += cur_valid_loss
                    # acc
                    pred = torch.sigmoid(outputs)
                    pred = torch.round(pred)


                    for t, p in zip(labels.view(-1), pred.view(-1)):
                        confusion_matrix_val[t.long(), p.long()] += 1

            precision_val, recall_val, f1_val, accuracy_val = getMetrics(confusion_matrix_val)


            scheduler.step()

            train_loss = train_loss / len(train_loader)
  
            train_acc = accuracy_train
            val_loss = val_loss / len(val_loader)
   
            val_acc = accuracy_val

            print(f"Epoch:{epoch + 1} / {epochs},lr: {optimizer.param_groups[0]['lr']:.5f} train loss:{train_loss:.5f},train acc: {train_acc:.5f},valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_precision'].append(precision_train)
            history['train_recall'].append(recall_train)
            history['train_f1'].append(f1_train)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(precision_val)
            history['val_recall'].append(recall_val)
            history['val_f1'].append(f1_val)
        
        # for each epoch mean in one fold assign to single value
        for key in history.keys():
            foldperf[key].append(np.mean(torch.tensor(history[key], device = 'cpu').numpy()))


    print('\n\nPerformance of {} fold cross validation'.format(folds))
    print("average train results")
    print("loss      :",np.mean(foldperf["train_loss"]))
    print("acc       :",np.mean(foldperf["train_acc"]))
    print("precision :",np.mean(foldperf["train_precision"]))
    print("recall    :",np.mean(foldperf["train_recall"]))
    print("f1 score  :",np.mean(foldperf["train_f1"]))
    
    
    
    print("average val results")
    print("loss      :",np.mean(foldperf["val_loss"]))
    print("acc       :",np.mean(foldperf["val_acc"]))
    print("precision :",np.mean(foldperf["val_precision"]))
    print("recall    :",np.mean(foldperf["val_recall"]))
    print("f1 score  :",np.mean(foldperf["val_f1"]))
    

    test_acc = 0
    confusion_matrix_test = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.float()
            images, labels = images.to(device), labels.to(device)

            
            outputs = model(images)

           
            pred = torch.sigmoid(outputs)
            pred = torch.round(pred)

            test_acc += (pred == labels).sum().item()
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix_test[t.long(), p.long()] += 1

    precision_test, recall_test, f1_test, accuracy_test = getMetrics(confusion_matrix_test)


    print(f'\n\nTest Accuracy:', accuracy_test, 'Test Precision:', precision_test, 'Test Recall:', recall_test, 'Test F1:', f1_test)

    return foldperf

def ploting(hist,folds):
    
    fig = make_subplots(rows=1, cols=2,subplot_titles=['loss','accuracy'])
    fig.add_trace(go.Scatter(x=[*range(1,folds+1)], y=hist["train_loss"],name='train loss'),row=1, col=1)
    fig.add_trace(go.Scatter(x=[*range(1,folds+1)], y=hist["val_loss"],name='val loss'),row=1, col=1)
    fig.add_trace(go.Scatter(x=[*range(1,folds+1)], y=hist["train_acc"],name='train acc'),row=1, col=2)
    fig.add_trace(go.Scatter(x=[*range(1,folds+1)], y=hist["val_acc"],name='val acc'),row=1, col=2)
    fig.update_layout(template='plotly_white');fig.update_layout(margin={"r":0,"t":60,"l":0,"b":0},height=300)
    fig.show()

VGG_model = models.vgg16(pretrained=True)

for name, param in VGG_model.named_parameters():
    param.requires_grad = False

# define out classifier
binary_classifier = nn.Sequential(
   nn.Linear(in_features=25088, out_features=2048),
   nn.ReLU(),
   nn.Linear(in_features=2048, out_features=1024),
   nn.ReLU(),
   nn.Linear(in_features=1024, out_features=512),
   nn.ReLU(),
   nn.Linear(in_features=512, out_features=1)
)

# replace model class classifier attribute:
VGG_model.classifier = binary_classifier

VGG_model=VGG_model.to(device)
summary(VGG_model,(3,200,200),32)
params7 = {
    "model": VGG_model,
    "train":final_train_dataset,
    "test":test_dataset,
    "device":device ,
    "lr" :0.0002,
    "batch_size":32,
    "epochs": 15,
    "gamma": 0.5 ,
    "patience": 7,
    "folds": 5,
    "l2":0.09
}

hist7 = train_model(params7)

ploting(hist7,5)