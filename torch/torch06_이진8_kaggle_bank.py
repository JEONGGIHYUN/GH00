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
path = './_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'train.csv', index_col=[0,1,2])
# print(train_csv) # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=[0,1,2]) 
# print(test_csv) #  [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) 
# print(submission_csv) # [110023 rows x 1 columns]

# print(train_csv.shape) # (165034, 13)
# print(test_csv.shape) # (110023, 12)
# print(submission_csv.shape) # (110023, 1)

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],
#       dtype='object')
'''
df = pd.DataFrame(train_csv)

# train_csv = train_csv['Geography'].str.replace('France', '1')
df = df.replace({'Geography':'France'}, '0')
df = df.replace({'Geography':'Germany'}, '1')
df = df.replace({'Geography':'Spain'}, '2')

df = df.replace({'derGen':'Male'}, '1')
df = df.replace({'derGen':'Female'}, '0')
'''
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
derGen_mapping = {'Male': 1, 'Female': 0}

train_csv['Geography'] = train_csv['Geography'].map(geography_mapping)
train_csv['Gender'] = train_csv['Gender'].map(derGen_mapping)

test_csv['Geography'] = test_csv['Geography'].map(geography_mapping)
test_csv['Gender'] = test_csv['Gender'].map(derGen_mapping)


x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=4524)


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

# y_train = torch.FloatTensor(y_train).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)

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
    nn.Linear(10, 64),
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