from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import time


ds=load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets",
                split='train')

print("done loading dataset")
ds=ds.train_test_split(test_size=0.2,seed=42)

x_train=ds['train']['image']
y_train=ds['train']['label']
x_test=ds['test']['image']
y_test=ds['test']['label']

print("done splitting dataset")

class CNNModel(nn.Module):
    def __init__(self,input_param):
        super(CNNModel,self).__init__()
        self.conv_layer=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2)
        )
        self.dense_layer=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(-1,128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,2)
        )
        
    def forward(self,x):
        x=self.conv_layer(x)
        x.view(-1,10)
        x=self.dense_layer(x)
        return x

def gen(batch_size=32):
    X=[]
    y=[]
    for i in ds:
        X.append(i['image'])
        y.append(i['label'])
        if len(X)==batch_size:
            yield np.stack(X),np.stack(y)
            X,y=[],[]
    



