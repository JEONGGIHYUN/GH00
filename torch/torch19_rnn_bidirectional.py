import numpy as np
import pandas as pd
import torch.optim as optim
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
        # self.cell = nn.RNN(input_size=1,    
                        #    hidden_size=32,  
                        #    num_layers=1,
                        #    batch_first=True,
                        #    )
        self.cell = nn.RNN(1, 32, batch_first=True,
                           bidirectional=True # 디폴트 : false
                           )
        self.fc1 = nn.Linear(3*32*2,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        # x,hidden_state = self.cell(x) 
        # x, h = self.cell(x)
        x,_ = self.cell(x)
        x = self.relu(x)
        
        # x = x.reshape(-1,3*32)
        x = x.contiguous()
        x = x.view(-1, 3*32*2)  # reshape보다 view가 더 좋을 수도 있으나 하드웨어의 성능이 좋아 크게 의미가 없을 수 있다. # view를 사용할때 contiguous가 권장된다.
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
model = RNN().to(DEVICE)

from torchsummary import summary

summary(model,(3,1))

#3 컴파일 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model,criterion, optimizer, loader):
    epoch_loss = 0
    
    model.train()
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch)
        loss.backward() # 가중치 계산
        optimizer.step() # 가중치 갱신
        
        epoch_loss += loss.item()
               
    return epoch_loss / len(loader)
        
def evaluate(model, criterion, loader):
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(loader)
    
epochs = 3000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    
    if epoch % 20==0:
        print('epoch{}, loss{}'.format(epoch,loss))
    
    
    # val_loss = evaluate(model, criterion, train_loader)
    
x_pred = np.array([[8,9,10]])
    
def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE) # (1,3) -> (1,3,1)
        
        y_pred = model(data)
        
    return y_pred.cpu().numpy()
        
# last_loss = evaluate(model, criterion, train_loader)
# print('최종 loss :', last_loss)
y_pred = predict(model, x_pred)
print('=================================================================')
print(y_pred)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    