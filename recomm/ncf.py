import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np

# Load the MovieLens dataset
data = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
print('len(data): ', len(data))
#data.to_csv('ff.csv')

# Drop the timestamp column
data = data.drop('timestamp', axis=1)

# Split the data into training and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Extract user and item IDs and ratings
train_user_ids = train['user_id'].values
train_item_ids = train['item_id'].values
train_ratings = train['rating'].values.astype('float32')

test_user_ids = test['user_id'].values
test_item_ids = test['item_id'].values
test_ratings = test['rating'].values.astype('float32')


# Number of unique users and items
num_users = data['user_id'].max()+1
num_items = data['item_id'].max()+1
print('num_users, num_items: ', num_users, num_items)
# Embedding dimension
embedding_dim = 32

# User embedding
user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
user_vector = Flatten(name='user_vector')(user_embedding)

# Item embedding
item_input = Input(shape=(1,), name='item_input')
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)
item_vector = Flatten(name='item_vector')(item_embedding)

# Concatenate user and item vectors
concat = Concatenate()([user_vector, item_vector])

# Fully connected layers
fc1 = Dense(128, activation='relu')(concat)
fc2 = Dense(64, activation='relu')(fc1)
output = Dense(1)(fc2)

# Build the model
model = Model([user_input, item_input], output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    [train_user_ids, train_item_ids],
    train_ratings,
    batch_size=64,
    epochs=10,
    validation_data=([test_user_ids, test_item_ids], test_ratings)
)


# Evaluate the model
test_loss = model.evaluate([test_user_ids, test_item_ids], test_ratings)
print(f'Test Loss: {test_loss}')

# Make predictions
user_id = 1  # Example user ID
item_id = 117  # Example item ID

# Convert user_id and item_id to NumPy arrays
#user_id_array = np.array([user_id])
#item_id_array = np.array([item_id])
user_id_array = np.array([196, 186, 22, 244, 166])
item_id_array = np.array([242, 302, 377, 51, 346])
predicted_rating = model.predict([user_id_array, item_id_array])
#print(f'Predicted rating for user {user_id} and item {item_id}:',predicted_rating[item_id][0])
print(predicted_rating)
