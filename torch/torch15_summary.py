import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

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
# train_dataset = MNIST(path, train=True, download=False)
# test_dataset = MNIST(path, train=False, download=False)

train_dataset = MNIST(path, train=True, download=False, transform=transf)
test_dataset = MNIST(path, train=False, download=False, transform=transf)

print(train_dataset[0][0].shape) # torch.Size([1, 56, 56]) 토치는 배치사이즈 채널이 다르다
print(train_dataset[0][1]) # 5

# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# scaler = MinMaxScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)
# x_train = train_dataset.data
# y_train = train_dataset.targets
# x_test = test_dataset.data
# y_test = test_dataset.targets

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets

# print(x_train.shape, x_test.size())
# print(y_test.shape, y_test.size())

### x_train/127.5 - 1 # 예의 값의 범위는? -1 ~ 1 # 정규화라기 보다는 표준화에 가깞다. (Z Score-정규화 )

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

model = CNN(1).to(DEVICE)

# model.summary() # 텐서플로는 이 코드를 사용

print(model)

from torchsummary import summary

summary(model,(1,56,56)) # TypeError: summary() missing 1 required positional argument: 'input_size'
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 54, 54]             640
#               ReLU-2           [-1, 64, 54, 54]               0
#          MaxPool2d-3           [-1, 64, 27, 27]               0
#            Dropout-4           [-1, 64, 27, 27]               0
#             Conv2d-5           [-1, 32, 25, 25]          18,464
#               ReLU-6           [-1, 32, 25, 25]               0
#          MaxPool2d-7           [-1, 32, 12, 12]               0
#            Dropout-8           [-1, 32, 12, 12]               0
#             Conv2d-9           [-1, 16, 10, 10]           4,624
#              ReLU-10           [-1, 16, 10, 10]               0
#         MaxPool2d-11             [-1, 16, 5, 5]               0
#           Dropout-12             [-1, 16, 5, 5]               0
#            Linear-13                   [-1, 16]           6,416
#            Linear-14                   [-1, 10]             170
# ================================================================
# Total params: 30,314
# Trainable params: 30,314
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 3.97
# Params size (MB): 0.12
# Estimated Total Size (MB): 4.09
# ----------------------------------------------------------------

































































































