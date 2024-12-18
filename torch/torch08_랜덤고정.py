
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
######################################################
# def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)

# set_seed(1004) 
######################################################
SEED = 1006

import random
random.seed(SEED) # 파이썬 랜덤 고정
np.random.seed(SEED) # 넘파이 랜덤 고정
## 토치 시드 고정
torch.manual_seed(SEED)
## 토치 쿠다 시드 고정
torch.cuda.manual_seed_all(SEED)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=SEED, stratify=y, shuffle=True)

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

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()
    optimizer.zero_grad() #그라디언트배니싱과 그라디언트 익스플로드 차이 깜지 쓰기
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward() # 기울기 계산
    optimizer.step() # 가중치 갱신
    return loss.item()

EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    # print('epoch :{}, loss :{:.8f}'.format(epoch, loss))
    print(f'epoch :{epoch}, loss :{loss:.8f}')


#4. 평가 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print('loss : ',loss)

######## acc 출력 #########
from sklearn.metrics import accuracy_score
# y_pred = model(x_test)
# print(y_pred[:5])
# # tensor([[  51.2528,   26.5833,  -93.8818],
#         # [  18.9561,   31.1490,  -64.0205],
#         # [  62.1401,   29.3186, -109.9416],
#         # [ -19.0389,   -3.1935,   22.5319],
#         # [ -15.8118,   -4.0514,   20.0309]], grad_fn=<SliceBackward0>)
y_pred = torch.argmax(model(x_test),dim=1)
# print(y_pred[:5])

# score = (y_pred == y_test).float().mean()
# print('accuracy : {:.4f}'.format(score))
# print(f'accuracy :{score:.4f}')
#######################################################################################
# y_pred = model(x_test)
# acc = accuracy_score(y_test.cpu().numpy(), np.argmax(y_pred.detach().cpu().numpy(),axis=1))
# print('acc : {:.4f}'.format(acc))

score2 = accuracy_score(y_test.cpu().numpy(),
                        y_pred.cpu().numpy())
print('accuracy_score :{:.4f}'.format(score2))
print(f'accuracy_score :{score2:.4f}')










