import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Concatenate, Dense, Input

# 예시 데이터 준비
ratings = pd.read_csv('ratings2.csv')

#train, test = train_test_split(ratings, test_size=0.1, random_state=42)

# 하이퍼파라미터 정의
n_users = ratings['user_id'].nunique()
n_items = ratings['item_id'].nunique()
n_factors = 50  # 잠재 요인(Latent Factors) 수

# NCF 모델 정의
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

# 학습 데이터 준비
train_user = ratings['user_id'].values
train_item = ratings['item_id'].values
train_rating = ratings['rating'].values

ncf_model.fit([train_user, train_item], train_rating, epochs=20, verbose=1)

# 추천 수행
user_id = 1  # 추천할 사용자 ID
item_ids = ratings['item_id'].unique()  # 모든 아이템 목록

# 예측 수행
predicted_ratings = ncf_model.predict([np.array([user_id]*len(item_ids)), np.array(item_ids)])

print(predicted_ratings)

# 상위 N개 아이템 추천
top_N = 5
top_items_indices = np.argsort(predicted_ratings[:, 0])[-top_N:][::-1]

# 실제 item_id 값으로 반환
recommended_items = item_ids[top_items_indices]
print("Recommended items:", recommended_items)

### only unseen

# 이미 본 아이템을 제외한 추천 수행
seen_items = ratings[ratings['user_id'] == user_id]['item_id'].values
unseen_item_ids = [item for item in item_ids if item not in seen_items]

# 예측 수행
predicted_ratings2 = ncf_model.predict([np.array([user_id]*len(unseen_item_ids)), np.array(unseen_item_ids)])

print(predicted_ratings2)

# 상위 N개 아이템 추천
top_N = 5
top_items_indices = np.argsort(predicted_ratings[:, 0])[-top_N:][::-1]

# 실제 item_id 값으로 반환
recommended_items2 = np.array(unseen_item_ids)[top_items_indices]
print("Recommended items (unseen):", recommended_items2)
