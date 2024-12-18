def find_max_index(arr):
    # 배열이 비어있는 경우 -1 반환
    if not arr:
        return -1
    
    max_index = 0  # 최댓값의 인덱스를 저장할 변수 초기화
    max_value = arr[0]  # 첫 번째 요소를 최댓값으로 설정
    
    # 배열의 각 요소를 순회하면서 최댓값과 그 인덱스를 찾음
    for i in range(1, len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_index = i
    
    return max_index

# 예제 사용
print(find_max_index([1, 3, 8, 7, 4, 5]))  # 출력: 2
print(find_max_index([1, 2, 3, 4, 5, 6]))  # 출력: 5