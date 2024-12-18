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
path = 'C:\\ai5\\_data\\dacon\\따릉이\\'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # [715 rows x 1 columns]
# print(submission_csv) # NaN으로 나오는 데이터는 없는 데이터를 뜻한다. 결측치 : 일반적인 데이터 집합에서 벗어난다는 뜻을 가진 이상치(outlier)의 하위 개념

# print(train_csv.shape) # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission_csv.shape) # (715, 1)

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='object')

# print(train_csv.info())

############ 결측치 처리 1. 삭제 ##############
train_csv.isnull().sum()
# print(train_csv.isna().sum())

train_csv = train_csv.dropna() 
# print(train_csv.isna().sum())
# print(train_csv) # [1328 rows x 10 columns]

# print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean()) # 결측치 채우기 
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
# print(x) # [1328 rows x 9 columns]

y = train_csv['count']

y = y.apply(pd.to_numeric)

# print(y.shape) # (1328, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4343)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# x_train = torch.DoubleTensor(x_train).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)

# y_train = torch.Tensor(y_train).to(DEVICE)
# y_test = torch.Tensor(y_test).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)

y_train = torch.FloatTensor(y_train.values).to(DEVICE)
y_test = torch.FloatTensor(y_test.values).to(DEVICE)

# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# y_train = torch.DoubleTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.DoubleTensor(y_test).unsqueeze(1).to(DEVICE)

# y_train = torch.IntTensor(y_train).to(DEVICE)
# y_test = torch.IntTensor(y_test).to(DEVICE)
# int - long
# float - double

# print('============================================================================')
# print(x_train.shape, x_test.shape) # 
# print(y_train.shape, y_test.shape) # 
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

model = Model(9, 1).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
    
    
    
epochs=2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs : {}, loss : {}'.format(epoch,loss))
    
    
print('=================================================')

#4.평가 예측
def evaluate(model, criterion,x, y):
    model.eval() # 평가모드 // 역전파, 가중치 갱신, 기울기 계산할수 있기도 없기도,
                 # 드롭아웃, 배치모델 <- 얘네들 몽땅 하지마!
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

############################################################################################################
from sklearn.metrics import r2_score
# result = model(x_test).detach().cpu().numpy()
# y_test_np = y_test.cpu().numpy()

# r2_score 계산
# acc = r2_score(y_test_np, np.round(result))
# print('acc는?', acc)
result = model(x_test)
acc = r2_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
print('acc는?', acc)
       
# 최종 loss : 2.632242441177368
# acc는? 0.9736842105263158
        
# y_pred = model(x_test)

# y_pred = np.round(y_pred)

# accuracy = accuracy_score(y_test, y_pred)

# print('accuracy_score :', accuracy.detach().cpu().numpy())