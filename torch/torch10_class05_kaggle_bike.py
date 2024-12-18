import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__,'사용DEVICE :', DEVICE)

#1. 데이터
path = 'C:\\ai5\\_data\\bike-sharing-demand\\'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(sampleSubmission.shape) # (6493, 1)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) 

######### 결측치 확인 #########
print(train_csv.isnull().sum())
print(train_csv.isna().sum())

print(test_csv.isnull().sum())
print(test_csv.isna().sum())

###### x와 y를 분리 ########
x = train_csv.drop(['casual','registered','count'], axis=1)
print(x.shape) # (10886, 8)
# print(x)

y = train_csv['count']
print(y.shape)

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=10)

################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

x_train = torch.FloatTensor(x_train.values).to(DEVICE)
x_test = torch.FloatTensor(x_test.values).to(DEVICE)

# x_train = torch.DoubleTensor(x_train).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)


# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)

y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

# y_train = torch.DoubleTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.DoubleTensor(y_test).unsqueeze(1).to(DEVICE)

# y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)
# int - long
# float - double

# print('============================================================================')
# print(x_train.shape, x_test.shape) # torch.Size([455, 30]) torch.Size([114, 30])
# print(y_train.shape, y_test.shape) # torch.Size([455, 1]) torch.Size([114, 1])
# print(type(x_train), type(y_train)) # <class 'torch.Tensor'> <class 'torch.Tensor'>



#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.Linear(64, 100),
#     nn.ReLU(),
#     nn.Linear(100, 32),
#     nn.ReLU(),
#     nn.Linear(32, 10),
#     nn.ReLU(),
#     nn.Linear(10, 1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self,input_dim, output_dim):
        # super().__init__() # 명시 해도 되고 안해도 된다 디폴트 값 
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    #순전파 (실행시키는 인스턴트) (클래스 안에서 쓰는 함수는 메서드 이다. (함수 = 메서드(method)))
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.linear5(x)
        # x = self.sigmoid(x)
        return x

model = Model(8, 1).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
    
    
    
epochs=5000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs : {}, loss : {}'.format(epoch,loss))
    
    
print('=================================================')

#4.평가 예측
def evaluate(model, criterion,x_test, y_test):
    model.eval() # 평가모드 // 역전파, 가중치 갱신, 기울기 계산할수 있기도 없기도,
                 # 드롭아웃, 배치모델 <- 얘네들 몽땅 하지마!
    with torch.no_grad():
        y_pred = model(x_test)
        loss2 = criterion(y_test, y_pred)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

############################################################################################################
from sklearn.metrics import r2_score

result = model(x_test)
acc = r2_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
print('r2는?', acc)
       
# 최종 loss : 2.632242441177368
# acc는? 0.9736842105263158
        
# y_pred = model(x_test)

# y_pred = np.round(y_pred)

# accuracy = accuracy_score(y_test, y_pred)

# print('accuracy_score :', accuracy.detach().cpu().numpy())