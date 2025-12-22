from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
import time

ds=load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets",
                split='train')

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
            nn.Linear(_,128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,2)
        )
        
model=CNNModel(_)

def gen(batch_size=32):
    X=[]
    y=[]
    for i in ds:
        X.append(i['image'])
        y.append(i['label'])
        if len(X)==batch_size:
            yield np.stack(X),np.stack(y)
            X,y=[],[]
    



