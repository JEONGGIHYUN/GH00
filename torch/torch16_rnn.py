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
        self.cell = nn.RNN(input_size=1,    # 피쳐갯수
                           hidden_size=32,  # 아웃풋 노드의 갯수
                        #    num_layers=1,  # 디폴트
                           batch_first=True, # 그냥 연산하면 3,n,32 으로 나오기 때문에 batch_first='True'로 설정하여 n,3,32 로 원래대로 바꿔줘야 한다.
                           ) # (3,n,32) -> batch_first=True -> (n, 3, 32)
        self.fc1 = nn.Linear(3*32,16) # (n,3*32) -> (n,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,3)
        self.relu = nn.ReLU()

    def forward(self,x):
        # x,hidden_state = self.cell(x) # 출력을 두 개로 명시하여야 한다 아웃풋 을 hidden_state를 출력하지 않고 이용하려면 x,_이런 식으로 지정해주면 된다.
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
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#                RNN-1  [[-1, 3, 32], [-1, 2, 32]]               0
#               ReLU-2                [-1, 3, 32]               0
#             Linear-3                   [-1, 16]           1,552
#             Linear-4                    [-1, 8]             136
#               ReLU-5                    [-1, 8]               0
#             Linear-6                    [-1, 3]              27
# ================================================================
# Total params: 1,715
# Trainable params: 1,715
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.05
# Params size (MB): 0.01
# Estimated Total Size (MB): 0.05
# ----------------------------------------------------------------















































































































