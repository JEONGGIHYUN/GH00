import pandas as pd
print(pd.__version__) # 1.3.4

data = [
    ["삼성", "1000", "2000"],
    ["현대", "1100", "3000"],
    ["LG", "2000", "500"],
    ["아모레", "3500", "6000"],
    ["네이버", "100", "1500"],
]

index = ['031', '059', '033', '045', '023']
columns = ["종목명", "시가", "종가"]

df = pd.DataFrame(data=data, index=index, columns=columns)

print(df)
#      종목명    시가    종가
# 031   삼성   1000  2000
# 059   현대   1100  3000
# 033   LG     2000   500
# 045  아모레   3500  6000
# 023  네이버    100  1500

print("============================= iloc 아모레와 네이버의 시가 ======================================")
print(df.iloc[3:5, 1])  # 행, 열
print(df.iloc[3:, 1])
# 045    3500
# 023     100
# Name: 시가, dtype: object
# -> 2개 다 똑같이 나옴.

print(df.iloc[[3, 4], 1])   
# 045    3500
# 023     100
# Name: 시가, dtype: object

print(df.iloc[[2, 4], 1])   # LG와 네이버의 시가 // 특정 행만 뽑기

# print(df.iloc[3:, '시가']) # 에러 ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types
# print(df.iloc[[2, 4], '시가']) # ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types

print("============================= loc 아모레와 네이버의 시가 ======================================")
# print(df.loc[3:5, 1])  # KeyError: 1
# print(df.loc["045" : "023", 1]) # KeyError: 1

print(df.loc["045" : "023", "시가"])
# 045    3500
# 023     100
# Name: 시가, dtype: object

print(df.loc["045":, '시가'])
# 045    3500
# 023     100
# Name: 시가, dtype: object

print(df.loc[["033", "023"], "시가"])
# 033    2000
# 023     100
# Name: 시가, dtype: object

# print(df.loc[1:, '시가'])   # TypeError: cannot do slice indexing on Index with these indexers [1] of type int

# print(df.loc["045":, 1])    # KeyError: 1

# print(df.loc[["033", "023"], 1])    # KeyError: 1

print("================================================")
###### 까묵어라!!!! 까묵어라!!! 까묵어라!!!
# print(df.loc['045':][2])             # KeyError: 2

# print(df.loc['045', '033'][2])  # KeyError: '033'

# print(df.loc[['033', '045']].iloc[2]) # IndexError: single positional indexer is out-of-bounds

# print(df.loc['033'].iloc[2])

# print(df.loc['033':'023'].iloc[2])  

# print(df.loc[['033':'023']].iloc[2])    # 에러

# print(df.iloc[[2, 4]][2]) # KeyError: 2

# print(df.iloc[2].loc['시가'])   # 2000

# print(df.iloc[2].loc['시가'])   # KeyError: '시가'
# print(df.iloc[2:3].loc['시가']) # KeyError: '시가'
# print(df.iloc[2:3].iloc[2])  # IndexError: single positional indexer is out-of-bounds
 
print(df.iloc[1:4].iloc[1:4])   # 행뽑고, 그 뽑은 행에서 또 행뽑고....
#      종목명    시가    종가
# 033   LG  2000   500
# 045  아모레  3500  6000

