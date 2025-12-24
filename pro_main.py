from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
import time
from torchinfo import summary
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train=transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
transform_test=transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
data_prep_s=time.time()

ds=load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets",split='train')
print("Dataset loaded successfully.")

ds=ds.train_test_split(test_size=0.3,seed=42)
train=ds['train']
test=ds['test']

train=train.with_transform(
    lambda x:{
        "image":[transform_train(img) for img in x["image"]],
        "label":x["label"]
    }
)
test=test.with_transform(
    lambda x:{
        "image":[transform_test(img) for img in x['image']],
        "label":x["label"]
    }
)
data_prep_e=time.time()
print(f"Data preprocessing completed in {data_prep_e - data_prep_s:.2f} seconds.")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.conv_layer=nn.Sequential(
            nn.Conv2d(3,32,3,stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32,64,3,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten=nn.Flatten()
        self.dense_layer=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10816,128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x=self.conv_layer(x)
        x=self.flatten(x)
        x=self.dense_layer(x)
        return x

def gen(data,batch_size=256):
    X=[]
    y=[]
    for i in data:
        X.append(i["image"].to(device))
        y.append(i["label"])
        if len(X)==batch_size:
            yield torch.stack(X), torch.tensor(y,dtype=torch.long).to(device)
            X,y=[],[]

model=CNNModel().to(device)
epochs=10
batch=256
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs*(len(train)//batch)
)
model.train()
countx=0
model_train_start=time.time()
train_loss=[]

for i in range(epochs):
    train=train.shuffle(seed=i)
    for x_batch,y_batch in gen(train,batch):
        optimizer.zero_grad()
        output=model(x_batch)
        loss=criterion(output,y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        countx+=batch
        print(f"Epoch={i+1}/{epochs}, Samples processed: {countx}, Loss: {loss.item()}")
    countx=0
    train_loss.append(loss.item())

model_train_end=time.time()
print(f"Training completed in {model_train_end - model_train_start:.2f} seconds.")

plt.plot(range(1,epochs+1),train_loss)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Epochs")
plt.savefig("training_loss_adam.png")
plt.close()

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

summary(model,input_size=(1,3,224,224))
torch.save(model.state_dict(),"cnn-beta-adam-cosine.pth")
print("Model saved as cnn-beta-adam-cosine.pth")