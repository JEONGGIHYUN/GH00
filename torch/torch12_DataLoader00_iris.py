import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__,'사용DEVICE :', DEVICE)

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)

# print(x.shape, y.shape) # torch.Size([150, 4]) torch.Size([150])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=16, stratify=y, shuffle=True)

# print(x_train.size(), y_train.size()) # torch.Size([120, 4]) torch.Size([120])
# print(x_test.size(), y_test.size()) # torch.Size([30, 4]) torch.Size([30])

# x_train = torch.FloatTensor(x_train)

# y_train = torch.LongTensor(y_train).to(DEVICE)

# x_test = torch.FloatTensor(x_test)

# y_test = torch.LongTensor(y_test).to(DEVICE)
# print(x_train.shape, y_train.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

x_train = x_train.to(DEVICE)
x_test = x_test.to(DEVICE)
y_train = y_train.to(DEVICE)
y_test = y_test.to(DEVICE)

print(x_train.size(), y_train.size())
print(x_test.size(), y_test.size()) 

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


#2. 모델
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3),            
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

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

# ######## acc 출력 #########
# from sklearn.metrics import accuracy_score

# y_pred = torch.argmax(model(test_loader[0].np()),dim=1)

# score2 = accuracy_score(test_loader[1].np().cpu().numpy(),
#                         y_pred.cpu().numpy())
# print('accuracy_score :{:.4f}'.format(score2))
# print(f'accuracy_score :{score2:.4f}')

######## acc 출력 ######### 2
from sklearn.metrics import accuracy_score

y_pred = torch.argmax(model(x_test),dim=1)

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_pred.cpu().numpy())
print('accuracy_score :{:.4f}'.format(score2))
print(f'accuracy_score :{score2:.4f}')

