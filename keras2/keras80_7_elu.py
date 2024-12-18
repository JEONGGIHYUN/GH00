import numpy as np
import matplotlib.pyplot as plt



x = np.arange(-5, 5, 0.1)

def elu(x, alpha=1.0):
    """
    Exponential Linear Unit (ELU) 활성화 함수.

    Parameters:
    - x: 입력값 (숫자 또는 배열)
    - alpha: 음수 입력에 대한 곡률 제어 매개변수 (기본값: 1.0)

    Returns:
    - ELU 결과값
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

elu = lambda x, alpha=1.0 : np.where(x > 0, x, alpha * (np.exp(x) - 1))

y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()