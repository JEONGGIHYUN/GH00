import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용 DEVICE : ', DEVICE)
# torch :  2.4.1+cu124  사용 DEVICE :  cuda


#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# print(x.shape) # torch.Size([3])
# print(x.size()) # torch.Size([3])

# x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
# print(x)
# tensor([[1.],
        # [2.],
        # [3.]])
x1 = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
x = (x1 - torch.mean(x1)) / torch.std(x1)
# print(x)
# tensor([[-1.],
        # [ 0.],
        # [ 1.]])

# print(x)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(x.shape,y.shape)
print(x.size(), y.size())
# torch.Size([3, 1]) torch.Size([3, 1])
# torch.Size([3, 1]) torch.Size([3, 1])


#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))

model = nn.Linear(1, 1).to(DEVICE) # 인풋, 아웃풋 # y = xw + b

#3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()          # 훈련모드
    optimizer.zero_grad()  # 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결.

    hypothesis = model(x)  # y = wx + b
    
    loss = criterion(hypothesis, y) # loss=mse()
    
    loss.backward() # 기울기(gradient)값 계산까지. # 역전파 시작 loss를 weight로 미분한 값이다.
    optimizer.step()    # 가중치(w) 갱신            # 역전파 끝
    
    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model,criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))    #verbose
    
#4. 평가, 예측
#loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()    # 평가모드
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)
    
x_pre = (torch.Tensor([[4]]).to(DEVICE) - torch.mean(x1)) / torch.std(x1)
result = model(x_pre)
### 스케일링 해야지 6나올지도
print('4의 예측값 :', result.item())
    
# 최종 loss : 1.640655113988032e-07
# 4의 예측값 : 3.9991872310638428





