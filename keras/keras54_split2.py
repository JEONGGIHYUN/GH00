import numpy as np
a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T #.reshape(10,2)
size = 5
# print(a)


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)



    
bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1, 0]
print(x, x.shape)
print(y, y.shape)




