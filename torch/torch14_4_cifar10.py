import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch: ', torch.__version__, '사용DEVICE : ', DEVICE)

import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5), (0.5))]) # nomalize((평균),(표준평균))
# min max (x) - 평균(0.5)고정 / 표편(0.5)고정
# minmax(xy_train) - 평균(0.5)# 고정
#------------------------------------------- = Z_Score Normalization (정규화와 표준화의 볶음밥)
#             표편(0.5)# 고정

#1. 데이터
path = './study/torch/_data/'
train_dataset = CIFAR10(path, train=True, download=True, transform=transf)
test_dataset = CIFAR10(path, train=False, download=True, transform=transf)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

####################### 잘 받아졌는지 확인 ##########################
# bbb = iter(train_loader)
# aaa = next(bbb)

# print(aaa)
# print(aaa[0].shape)
# print(len(train_loader)) # 1875 = 60009 / 32


#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1), # (n,64,54,54)
            # model.Conv2D(64, (3,3), stride=1, input_shape=(56, 56, 1))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # (n, 64, 27, 27)
            nn.Dropout(0.5),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), # (n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # (n, 32, 12, 12)
            nn.Dropout(0.5),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), # (n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # (n, 16, 5, 5)
            nn.Dropout(0.5),
        )        
        
        self.hidden_layer4 = nn.Linear(16*5*5, 16)
        self.output_layer = nn.Linear(in_features=16, out_features=10)
        
    def forward(self,x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)
        # x = flatten() # 케라스에선 위에것을 이렇게 사용했다.
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x

model = CNN(3).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4) # 0.0001

def train(model, criterion, optimizer, loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        y_pred = torch.argmax(hypothesis, 1)
        acc = (y_pred == y_batch).float().mean()
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, criterion, loader):
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            
            y_pred = torch.argmax(hypothesis, 1)
            acc = (y_pred == y_batch).float().mean()
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)
# loss, acc = model.evaluate(x_test, y_test)

epochs = 10
for epoch in range(1, epochs + 1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch:{}, loss:{:.4f}, acc:{:.3f}, val_loss:{:.4f}, val_acc{:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc
    ))

#4.평가 예측

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss :', last_loss)

