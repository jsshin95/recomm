import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 예시 데이터 준비
"""
ratings = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
    'item_id': [1, 2, 3, 1, 2, 2, 3, 4],
    'rating': [5, 3, 4, 4, 2, 2, 5, 3]
})
"""
ratings = pd.read_csv('ratings2.csv')
print(ratings)
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

#----------모델 정의 (MF)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dot, Input

# 하이퍼파라미터 정의
n_users = ratings['user_id'].nunique()
n_items = ratings['item_id'].nunique()
n_factors = 50  # 잠재 요인(Latent Factors) 수

# 모델 정의
user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=n_users+1, output_dim=n_factors, input_length=1)(user_input)
user_vec = Flatten()(user_embedding)

item_input = Input(shape=(1,))
item_embedding = Embedding(input_dim=n_items+1, output_dim=n_factors, input_length=1)(item_input)
item_vec = Flatten()(item_embedding)

dot_product = Dot(axes=1)([user_vec, item_vec])

model = Model([user_input, item_input], dot_product)
model.compile(optimizer='adam', loss='mse')


#----------모델 학습
# 학습 데이터 준비
train_user = train['user_id'].values
train_item = train['item_id'].values
train_rating = train['rating'].values

model.fit([train_user, train_item], train_rating, epochs=10, verbose=1)

#------------추천
user_id = 1  # 추천할 사용자 ID
item_ids = ratings['item_id'].unique()  # 모든 아이템 목록

# 예측 수행
predicted_ratings = model.predict([np.array([user_id]*len(item_ids)), np.array(item_ids)])
print(predicted_ratings)

# 상위 N개 아이템 추천
top_N = 5
recommended_items = np.argsort(predicted_ratings[:, 0])[-top_N:]
print("Recommended items:", recommended_items)

"""
#-----------NCF
from tensorflow.keras.layers import Concatenate, Dense

# 모델 정의
user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=n_users+1, output_dim=n_factors, input_length=1)(user_input)
user_vec = Flatten()(user_embedding)

item_input = Input(shape=(1,))
item_embedding = Embedding(input_dim=n_items+1, output_dim=n_factors, input_length=1)(item_input)
item_vec = Flatten()(item_embedding)

concat = Concatenate()([user_vec, item_vec])
dense = Dense(128, activation='relu')(concat)
output = Dense(1)(dense)

ncf_model = Model([user_input, item_input], output)
ncf_model.compile(optimizer='adam', loss='mse')
"""