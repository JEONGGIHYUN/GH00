import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__,'사용DEVICE :', DEVICE)

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1354,train_size=0.8, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
##############################################################################
from torch.utils.data import TensorDataset  # x,y를 합친다.
from torch.utils.data import DataLoader     # batch 정의.

train_set = TensorDataset(x_train, y_train) # 튜플 형태로 합쳐진다.
test_set = TensorDataset(x_test, y_test)
print(train_set)    # <torch.utils.data.dataset.TensorDataset object at 0x0000019FBBCA00B0>
print(type(train_set)) # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set)) # 455
print(train_set[0])  # 첫번째 x                    튜플과 리스트의 차이점
print(train_set[0][1]) #첫번째 y train_set[397] 까지 있다.

# 토치데이터셋 만들기 2. batch를 넣어준다.
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)
##############################################################################
#모델
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

def train(model,criterion,optimizer,loader):
    optimizer.zero_grad()
    total_loss = 0
    
    for x_batch,y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
    
    
    
epochs=50
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs : {}, loss : {}'.format(epoch,loss))
    
    
print('=================================================')

#4.평가 예측
def evaluate(model, criterion,loader):
    model.eval() # 평가모드 // 역전파, 가중치 갱신, 기울기 계산할수 있기도 없기도,
                 # 드롭아웃, 배치모델 <- 얘네들 몽땅 하지마!
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_pred = model(x_batch)
            loss2 = criterion(y_batch, y_pred)
            total_loss += loss2.item()
    return loss2 / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss :', last_loss)

############################################################################################################
from sklearn.metrics import r2_score

result = model(x_test)
acc = r2_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
print('acc는?', acc)
       