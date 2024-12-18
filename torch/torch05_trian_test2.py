import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용 DEVICE : ', DEVICE)
# torch :  2.4.1+cu124  사용 DEVICE :  cuda


#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))


x_pred = np.array([12,13,14])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=426,train_size=0.8)


x_pred = torch.FloatTensor(x_pred).unsqueeze(1).to(DEVICE)

model = nn.Sequential(nn.Linear(1, 100),
                      nn.Linear(100, 70),
                      nn.Linear(70, 50),
                      nn.Linear(50, 5),
                      nn.Linear(5, 3),
                      nn.Linear(3, 1),
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
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch,loss))
    
#4. 평가 예측

def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss:', loss2)

# x_pred = np.array([12,13,14])

results = model((x_pred).to(DEVICE))

print('12과 13과 14의 예측값 :', results.detach().cpu().numpy())