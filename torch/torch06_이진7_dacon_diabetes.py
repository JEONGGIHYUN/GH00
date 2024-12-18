import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__,'사용DEVICE :', DEVICE)
###################################################################
#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)

# print(train_csv.shape) # (652, 9)
# print(test_csv.shape) # (116, 8)
# print(submission_csv.shape) # (116, 1)

# print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    #    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
    #   dtype='object')

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=3)

# print(x_train.shape, x_test.shape) # (554, 8) (98, 8)

# x_train = x_train.to_numpy()
# x_test = x_test.to_numpy()

# x_train = x_train.reshape(554,2,2,2)
# x_test = x_test.reshape(98,2,2,2)

###################################################################
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# x_train = torch.DoubleTensor(x_train).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)


# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

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
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.Linear(64, 100),
    nn.ReLU(),
    nn.Linear(100, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model,critieropn,optimizer,x,y):
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
from sklearn.metrics import accuracy_score

result = model(x_test)
acc = accuracy_score(y_test.cpu().numpy(), np.round(result.detach().cpu().numpy()))
print('acc는?', acc)