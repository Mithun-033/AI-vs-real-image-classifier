
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
import time
start=time.time()
ds=load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets",
                split='train',
                streaming=True)


def gen(batch_size=32):
    X=[]
    y=[]
    for i in ds:
        X.append(i['image'])
        y.append(i['label'])
        if len(X)==batch_size:
            yield np.stack(X),np.stack(y)
            X,y=[],[]
    
s=next(iter(ds))
img=np.asarray(s["image"])
print("Image shape:",img.shape)
end=time.time()
print(end - start)
model=nn.Sequential(

)


