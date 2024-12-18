import numpy as np
import matplotlib.pyplot as plt



x = np.arange(-5, 5, 0.1)

def silu(x):
   return x * 1 / (1 + np.exp(-x))

silu = lambda x : x * 1 / (1 + np.exp(-x))

y = silu(x)

plt.plot(x,y)
plt.grid()
plt.show()

# swish는 relu보다 계산량이 많아서 모델이 커질수록 부답스럽다.