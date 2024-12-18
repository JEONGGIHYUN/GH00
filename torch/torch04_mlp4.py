import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용DEVICE :', DEVICE)

#1. 데이터
x = np.array([range(10)]).transpose()
y = np.array([[1,2,3,4,5,6,7,8,9,10]
              ,[10,9,8,7,6,5,4,3,2,1]
              ,[9,8,7,6,5,4,3,2,1,0]]).transpose()
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

model = nn.Sequential(nn.Linear(1, 100),
                      nn.Linear(100, 70),
                      nn.Linear(70, 50),
                      nn.Linear(50, 5),
                      nn.Linear(5, 3),
                      nn.Linear(3, 3),
                      ).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.MSELoss()

optimizer = optim.Adamax(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    
    optimizer.zero_grad()
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch,loss))
    
#4. 평가 예측

def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss:', loss2)

results = model(torch.Tensor([[11]]).to(DEVICE))

print('11과 0과 -1의 예측값 :', results.detach().cpu().numpy())












































































