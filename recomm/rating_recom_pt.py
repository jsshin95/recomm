import pandas as pd

df = pd.read_csv('ratings2.csv')

# X = (userID, movieID)
X = df[['userId', 'movieId']].values
Y = df[['rating']].values

# 9:1로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X, Y, test_size=0.1)

# X는 ID, 정수 -> int64
# Y는 실수 -> float32
train_dataset = TensorDataset(
    torch.tensor(train_X, dtype=torch.int64), torch.tensor(train_Y, dtype=torch.float32))
test_dataset = TensorDataset(
    torch.tensor(test_X, dtype=torch.int64), torch.tensor(test_Y, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=4, shuffle=False)