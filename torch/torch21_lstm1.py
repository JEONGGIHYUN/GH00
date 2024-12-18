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
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTM(1, 32, batch_first=True,
                           bidirectional=False # 디폴트 : false
                           )
        self.fc1 = nn.Linear(3*32,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        self.relu = nn.ReLU()

    def forward(self,x, h0=None, c0=None):
        # x,hidden_state = self.cell(x) 
        # x, h = self.cell(x)
        
        h0 = torch.zeros(1, x.size(0), 32).to(DEVICE) 
        c0 = torch.zeros(1, x.size(0), 32).to(DEVICE)         
        x,h0 = self.cell(x, [h0, c0])
        
        x = self.relu(x)
        
        
        # x = x.reshape(-1,3*32)
        x = x.contiguous()
        x = x.view(-1, 3*32)  #input_layer, hidden_layer, outputlayer()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
model = LSTM().to(DEVICE)

# from torchsummary import summary

# summary(model,(3,1))

#3 컴파일 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model,criterion, optimizer, loader):
    epoch_loss = 0
    
    model.train()
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)
        
        optimizer.zero_grad()
        
        h0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE)
        c0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE)
        
        hypothesis = model(x_batch, h0, c0)
        
        loss = criterion(hypothesis, y_batch)
        loss.backward() # 가중치 계산
        optimizer.step() # 가중치 갱신
        
        epoch_loss += loss.item()
               
    return epoch_loss / len(loader)
        
def evaluate(model, criterion, loader):
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)
            
            h0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE)
            c0 = torch.zeros(1, x_batch.size(0), 32).to(DEVICE)

            hypothesis = model(x_batch, h0, c0)
            
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(loader)
    
epochs = 3000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    
    if epoch % 20 == 0 :
        print('epoch{}, loss{}'.format(epoch,loss))
    
    
    # val_loss = evaluate(model, criterion, train_loader)
    
x_pred = np.array([[8,9,10]])
    
def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE) # (1,3) -> (1,3,1)
        
        
        h0 = torch.zeros(1, data.size(0), 32).to(DEVICE)
        c0 = torch.zeros(1, data.size(0), 32).to(DEVICE)        

        y_pred = model(data, h0, c0)
        
    return y_pred.cpu().numpy()
        
# last_loss = evaluate(model, criterion, train_loader)
# print('최종 loss :', last_loss)
y_pred = predict(model, x_pred)
print("=======================================")
print(y_pred)    # [[10.453035]]
print("=======================================")
print(y_pred[0]) # [10.453035]
print("=======================================")
print(f'{x_pred}의 예측값 : {y_pred[0][0]}') 