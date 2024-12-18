import numpy as np
import pandas as pd
import torch.optim as optum
import random 
import torch
import torch.nn as nn

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('torch: ', torch.__version__, '사용DEVICE : ', DEVICE)
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
print(DEVICE)

# Dense, SimpleRNN, LSTM,GRU

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3,],
              [2,3,4,],
              [3,4,5,],
              [4,5,6,],
              [5,6,7,],
              [6,7,8,],
              [7,8,9,],]
             )

y = np.array([4,5,6,7,8,9,10,])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0],x.shape[1], 1)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
print(x.shape, y.size())

from torch.utils.data import TensorDataset #x_y합치기
from torch.utils.data import DataLoader # batch정의

train_set = TensorDataset(x, y)

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# aaa = iter(train_loader)
# bbb = next(aaa) # aaa.next()
# print(bbb)

#2 모델
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.RNN(input_size=1,    
                           hidden_size=32,  
                           num_layers=1,  # 전설 : 디폴트 아니면 3,5 가 좋다는 속설이 있다. 
                           batch_first=True,
                           )
        self.fc1 = nn.Linear(3*32,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,3)
        self.relu = nn.ReLU()

    def forward(self,x):
        # x,hidden_state = self.cell(x) 
        # x, h = self.cell(x)
        x,_ = self.cell(x)
        x = self.relu(x)
        
        x = x.reshape(-1,3*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
model = RNN().to(DEVICE)

from torchsummary import summary

summary(model,(3,1))
