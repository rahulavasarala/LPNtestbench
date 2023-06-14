import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

#Step 0: select the device

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#Create the dataset loader

from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.from_numpy(x).to(torch.float32)
    self.y = torch.from_numpy(y).to(torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length
  
class LPNOracle:
    def __init__(self, secret, error_rate):
        self.secret = secret
        self.dimension = len(secret)
        self.error_rate = error_rate

    def sample(self, n_amount):
        # Create random matrix.
        A = np.random.randint(0, 2, size=(n_amount, self.dimension))
        # Add Bernoulli errors.
        e = np.random.binomial(1, self.error_rate, n_amount)
        # Compute the labels.
        b = np.mod(A @ self.secret + e, 2)
        return A, b
    
p = 0.125
dim = 12
s = np.random.randint(0, 2, dim)
lpn = LPNOracle(s, p)

A, b = lpn.sample(10000)
  
trainset = dataset(A, b)
print(trainset.__len__())
trainloader = DataLoader(trainset,batch_size=64,shuffle=False)

from torch import nn
from torch.nn import functional as F

class cryptnet(nn.Module):
  def __init__(self,input_shape):
    super(cryptnet,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x

chicken_net = cryptnet(trainset.x.shape[1])

learning_rate = 0.01
epochs = 700
optimizer = torch.optim.SGD(chicken_net.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()

#Now is the training part of the nueral network\

for i in range(epochs):
    for j,(x_train,y_train) in enumerate(trainloader):
        
        #calculate output
        output = chicken_net(x_train)
    
        #calculate loss
        loss = loss_fn(output,y_train.reshape(-1,1))

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("We reached epoch {a}".format(a = i))




