import pandas as pd
import numpy as np
import random 
import torch
import torch.nn as nn

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('torch: ', torch.__version__, '사용DEVICE : ', DEVICE)
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
print(DEVICE)

path = 'C:\\ai5\\_data\\kaggle\\netflix\\netflix-stock-prediction\\'

train_csv = pd.read_csv(path+'train.csv')
print(train_csv)
print(train_csv.info())
print(train_csv.describe())

import matplotlib.pyplot as plt
# data = train_csv.iloc[:,1:4] # index location 판다스는 열이 우선이다 그래서 넘파이나 파이썬 형태의 데이터를 스플릿하고 싶으면 loc iloc를 이용하여 넘파이와 파이썬 형태의 데이터로 똑같이 바꾼다.
# index location
# data['종가'] = train_csv['Close']
# print(data)

# hist = data.hist()
# plt.show()

# 데이터를 minmaxscale해 주었는데 데이터가 꼬여서 이상하게 출력이 된다.
# data = train_csv.iloc[:, 1:4]
# data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# data = pd.DataFrame(data)
# print(data.describe())

# 그래서 axis=0으로 지정하면 열에 맞춰서 계산을 해준다.
# data = train_csv.iloc[:, 1:4]
# data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# data = pd.DataFrame(data)
# print(data.describe())


from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader


class Custom_dataset(Dataset):
    def __init__(self):
        self.csv = train_csv
        
        self.x = self.csv.iloc[:,1:4].values # 싯가, 종가, 저가, 고가
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        #정규화
        
        self.y = self.csv['Close'].values     

    def __len__(self):
        return len(self.x) - 30
    
    def __getitem__(self, i):
        x = self.x[i:i+30]
        y = self.y[i+30]
        
        return x,y

aaa = Custom_dataset()
# print(aaa) # <__main__.Custom_dataset object at 0x000001C046E0D520>
# print(type(aaa)) # <class '__main__.Custom_dataset'>

# print(aaa[0])
# (array([[-80.90736342, -84.91016548, -79.90799031],
#        [-80.89786223, -84.90307329, -79.89830508],
#        [-80.90498812, -84.91252955, -79.90799031],
#        [-80.90736342, -84.91725768, -79.91283293],
#        [-80.91448931, -84.92434988, -79.91525424],
#        [-80.91448931, -84.91962175, -79.91283293],
#        [-80.91211401, -84.91962175, -79.91041162],
#        [-80.91448931, -84.92434988, -79.91767554],
#        [-80.91211401, -84.91725768, -79.91041162],
#        [-80.90973872, -84.91962175, -79.91283293],
#        [-80.91686461, -84.92434988, -79.91767554],
#        [-80.93349169, -84.94089835, -79.93946731],
#        [-80.93111639, -84.93853428, -79.937046  ],
#        [-80.94299287, -84.92198582, -79.93946731],
#        [-80.91686461, -84.91252955, -79.92251816],
#        [-80.91686461, -84.92198582, -79.92493947],
#        [-80.9263658 , -84.92434988, -79.92493947],
#        [-80.91686461, -84.92198582, -79.91525424],
#        [-80.9216152 , -84.93144208, -79.93946731],
#        [-80.94061758, -84.94326241, -79.94915254],
#        [-80.95011876, -84.95035461, -79.94673123],
#        [-80.93824228, -84.94089835, -79.937046  ],
#        [-80.93349169, -84.94089835, -79.95883777],
#        [-80.93586698, -84.94326241, -79.94673123],
#        [-80.94299287, -84.95271868, -79.95399516],
#        [-80.95486936, -84.95744681, -79.95399516],
#        [-80.95486936, -84.96217494, -79.96368039],
#        [-80.96199525, -84.96926714, -79.97336562],
#        [-80.96912114, -84.97635934, -79.97578692],
#        [-80.97387173, -84.9787234 , -79.97336562]]), 94)

# print(aaa[0][0].shape) # (30, 3)
# print(aaa[0][1]) # 94
# print(len(aaa)) # 937
# print(aaa[937]) # IndexError: index 967 is out of bounds for axis 0 with size 967

#  x는 (937, 30, 3), y는 (937, 1)

train_loader = DataLoader(aaa, batch_size=32)
# aaa = iter(train_loader)
# bbb = next(aaa) # aaa.next()
# print(bbb)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cell = nn.LSTM(3,64, batch_first=True,
                          bidirectional=False, 
                          )
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x,hiden_state = self.cell(x, h0, c0)
        
        # h0 = torch.zeros(1, x.size(0), 64).to(DEVICE) 
        # c0 = torch.zeros(1, x.size(0), 64).to(DEVICE)
        # x = torch.reshape(x, (x.shape[0], -1))  
        # x = x[:, -1, :]       
        x,hiden_state = self.cell(x,)        
        
        x = self.fc1(x)
        x = self.relu(x)        
        x = self.fc2(x)
        
        return x

model = LSTM().to(DEVICE)

# 컴파일 훈련
import torch.optim as optim
from torch.optim import Adam
criterion = nn.MSELoss()

optim = Adam(params=model.parameters(), lr=1e-4)

import tqdm # 프로그래스바를 돌리기 위해 사용

for epoch in range(1, 201):
    iterator = tqdm.tqdm(train_loader)
    # iterator = train_loader    
    for x, y in (iterator):
        optim.zero_grad()

        # h0 = torch.zeros(5, x.size(0), 64).to(DEVICE) # 넘레이어스 배치사이즈 히든사이즈 (5, 32, 64) h0를 안사용해도 자동으로 적용된다.
        # c0 = torch.zeros(5, x.size(0), 64).to(DEVICE) 
        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE))

        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))
        
        loss.backward()
        optim.step()
        
        iterator.set_description(f'epoch:{epoch} loss:{loss.item()}')
        
save_path = 'C:/ai5/_save/torch/'
torch.save(model.state_dict(),save_path + 't24.pth')


 