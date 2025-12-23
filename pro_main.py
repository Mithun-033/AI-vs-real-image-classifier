from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

ds=load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets",split='train')
print("Dataset loaded successfully.")

ds=ds.with_transform(
    lambda x:{
        "image":[transform(img) for img in x["image"]],
        "label":x["label"]
    }
)

ds=ds.train_test_split(test_size=0.2,seed=42)

train=ds['train']
test=ds['test']

print("Data preprocessing completed.")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.conv_layer=nn.Sequential(
            nn.Conv2d(3,32,3,stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,stride=2),
            nn.BatchNorm2d(128)
        )
        self.flatten=nn.Flatten()
        self.dense_layer=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128*27*27,32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,2)
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=self.flatten(x)
        x=self.dense_layer(x)
        return x

def gen(data,batch_size=64):
    X=[]
    y=[]
    for i in data:
        X.append(i["image"].to(device))
        y.append(i["label"])
        if len(X)==batch_size:
            yield torch.stack(X), torch.tensor(y,dtype=torch.long).to(device)
            X,y=[],[]

model=CNNModel().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.0000000001)
epoch=1
model.train()

for _ in range(epoch):
    for x_batch,y_batch in gen(train):
        optimizer.zero_grad()
        output=model(x_batch)
        loss=criterion(output,y_batch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {_+1}/{epoch}, Loss: {loss.item()}")

print("Training completed.")

correct=0
total=0
with torch.no_grad():
    model.eval()
    for x_batch,y_batch in gen(test):
        output=model(x_batch)
        pred=output.argmax(dim=1)
        correct+=(pred==y_batch).sum().item()
        total+=len(y_batch)
        print(f"Processed {total} samples.")

accuracy=correct/total
print(f"Test Accuracy: {accuracy*100:.2f}%")
