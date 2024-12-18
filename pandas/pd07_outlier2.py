import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
               [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

# 첫번째 컬럼과 두 번째 컬럼을 포문을 돌려서 아웃라이어가 빠지게 함수를 구성하기 

def outliers(data_out):
    outliers_info = {}
    num_columns = data_out.shape[1]

    for col_idx in range(num_columns):
        data_out = data_out[:, col_idx]

        quartile_1, q2, quartile_3 = np.percentile(data_out,    # percentile 백분율
                                                [25, 50, 75])
        
        print("1사분위 : ", quartile_1)  # 4.0
        print("q2 : ", q2)              # 7.0
        print("3사분위 : ", quartile_3)  # 10.0
        iqr = quartile_3 - quartile_1   # 10.0 - 4.0 = 6.0
        print("iqr : ", iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out>upper_bound) |
                        (data_out<lower_bound)), iqr

# def outliers(data_array):
#     """
#     numpy 배열에서 각 열의 이상치를 계산합니다.
    
#     Parameters:
#     data_array (numpy.ndarray): 2차원 numpy 배열
    
#     Returns:
#     dict: 각 열별로 이상치 정보 (이상치 인덱스, IQR, 상한선, 하한선)
#     """
#     outliers_info = {}
#     num_columns = data_array.shape[1]
    
#     for col_idx in range(num_columns):
#         data_out = data_array[:, col_idx]
        
#         quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
#         iqr = quartile_3 - quartile_1
#         lower_bound = quartile_1 - (iqr * 1.5)
#         upper_bound = quartile_3 + (iqr * 1.5)
        
#         # 이상치 인덱스
#         outliers = np.where((data_out > upper_bound) | (data_out < lower_bound))[0]
        
#         # 결과 저장
#         outliers_info[col_idx] = {
#             "outliers_indices": outliers,
#             "iqr": iqr,
#             "lower_bound": lower_bound,
#             "upper_bound": upper_bound,
#         }
        
#     return np.where((data_out>upper_bound) |
#                     (data_out<lower_bound)), iqr



outliers_loc, iqr = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.axhline(iqr, color='red', label='TQR')
# plt.show()

# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)