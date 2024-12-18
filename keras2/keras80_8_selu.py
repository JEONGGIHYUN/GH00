import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5, 5, 0.1)


def selu(x, alpha=1.67326, lambda_=1.0507):
    """
    Scaled Exponential Linear Unit (SELU) 활성화 함수.

    Parameters:
    - x: 입력값 (숫자 또는 배열)
    - alpha: SELU의 곡률 제어 매개변수 (기본값: 1.67326)
    - lambda_: SELU의 스케일링 매개변수 (기본값: 1.0507)

    Returns:
    - SELU 결과값
    """
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

selu = lambda x, alpha=1.67326, lambda_=1.0507 : lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()