import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용DEVICE :', DEVICE)

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5 ,1.6 ,1.5 ,1.4, 1.3]
              ]).transpose()
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5,6,7,7,9,10])
# print(x.shape, y.shape)
# 예측값 : [10, 1.3]

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

# 모델 구성

model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Linear(5, 10),
    nn.Linear(10, 20),
    nn.Linear(20, 10),
    nn.Linear(10, 5),
    nn.Linear(5, 1)).to(DEVICE)

# 컴파일 훈련

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(mode, criterion, optimizer, x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


epochs = 3000

for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x , y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
    
#4. 평가 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

results = model(torch.Tensor([[10,1.3]]).to(DEVICE))
print('10과 1.3의 예측값 :', results.item())





















































































