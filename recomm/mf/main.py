import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 파일 경로 정의
RATING_DATA_PATH = './data/ratings.csv'
# numpy 출력 옵션 설정
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def predict(Theta, X):
    """유저 취향과 상품 속성을 곱해서 예측 값을 계산하는 함수"""
    return Theta @ X


def cost(prediction, R):
    """행렬 인수분해 알고리즘의 손실을 계산해주는 함수"""
    return np.nansum((prediction - R)**2)


def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape
    
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)
    
    return Theta, X


def gradient_descent(R, Theta, X, iteration, alpha, lambda_):
    """Matrix Factorization gradient descent"""
    num_user, num_items = R.shape
    num_features = len(X)
    costs = []
        
    for _ in range(iteration):
        prediction = predict(Theta, X)
        error = prediction - R
        costs.append(cost(prediction, R))
                          
        for i in range(num_user):
            for j in range(num_items):
                if not np.isnan(R[i][j]):
                    for k in range(num_features):
                        # Update Theta, X
                        Theta[i][k] -= alpha * (np.nansum(error[i,:]*X[k,:])+lambda_*Theta[i][k])
                        X[k][j] -= alpha * (np.nansum(error[:,j]*Theta[:,k])+lambda_*X[k][j])
                        
    return Theta, X, costs


#----------------------테스트 코드----------------------
# 평점 데이터를 가지고 온다
ratings_df = pd.read_csv(RATING_DATA_PATH, index_col='user_id')

"""
lUser=[]
lItem=[]
lrating=[]
R0 = ratings_df.values
nUsers, nItems = R0.shape
for row in range(nUsers):
    for col in range(nItems):
        if R0[row][col] == R0[row][col]:
            lUser.append(row)
            lItem.append(col)
            lrating.append(R0[row][col])
            
ratings = pd.DataFrame({
    'user_id': lUser,
    'item_id': lItem,
    'rating': lrating
})
#print(ratings)
ratings.to_csv('ratings.csv', index=False)
"""
# 평점 데이터에 mean normalization을 적용한다
for row in ratings_df.values:
    row -= np.nanmean(row)
       
R = ratings_df.values
        
Theta, X = initialize(R, 10)  # 행렬 초기화
Theta, X, costs = gradient_descent(R, Theta, X, 200, 0.001, 0.01)  # 경사 하강
    
# 손실 시각화
#plt.plot(costs)
#plt.show()

fpred = predict(Theta, X)
print(fpred[1])

# 예제 배열 생성
arr = fpred[1]

# 상위 5개 값의 인덱스를 찾기
top_5_indices = np.argsort(arr)[-5:][::-1]

print("최대값 5개의 인덱스:", top_5_indices)
print("최대값 5개:", arr[top_5_indices])

"""
print('R')
print(R)
print('Theta @ X')
print(predict(Theta, X))
print('Error')
print(predict(Theta,X)-R)
print('cost')
print(cost(predict(Theta,X), R))
"""