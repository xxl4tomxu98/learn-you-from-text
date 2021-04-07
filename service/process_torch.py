import pickle, os, torch, torch.utils.data
import torch.optim as optim
import pandas as pd
from utils import paragraph_to_words, convert_and_pad_data, preprocess_data
from utils import prepare_imdb_data, build_dict, read_imdb_data
from model_torch import Sentiment
import numpy as np
from sklearn.metrics import accuracy_score


cache_dir = os.path.join("./static/cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists
data_dir = './static/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)

# Preprocess data
data, labels = read_imdb_data()
train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
word_dict = build_dict(train_X)

# save word_dict
with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)

# convert data sets into correct format for pytorch
train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)

# save the processed training dataset locally as train.csv file
pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

# save the processed testing dataset locally as test.csv file
pd.concat([pd.DataFrame(test_y), pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

# Fit the pytorch LSTM model for sentiment analysis, first load the saved train and test data   
train_sentiment = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=25000)
test_sentiment = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None, names=None, nrows=25000)

# Turn the input pandas dataframe into tensors
train_sentiment_y = torch.from_numpy(train_sentiment[[0]].values).float().squeeze()
train_sentiment_X = torch.from_numpy(train_sentiment.drop([0], axis=1).values).long()
test_sentiment_y = torch.from_numpy(test_sentiment[[0]].values).float().squeeze()
test_sentiment_X = torch.from_numpy(test_sentiment.drop([0], axis=1).values).long()

# Build the dataset
train_sentiment_ds = torch.utils.data.TensorDataset(train_sentiment_X, train_sentiment_y)
test_sentiment_ds = torch.utils.data.TensorDataset(test_sentiment_X, test_sentiment_y)

# Build the dataloader
train_sentiment_dl = torch.utils.data.DataLoader(train_sentiment_ds, batch_size=50)
test_sentiment_dl = torch.utils.data.DataLoader(test_sentiment_ds, batch_size=50)

# train the pytorch model
device = torch.device("cpu")
lstm_model = Sentiment(32, 100, 5000).to(device)
optimizer = optim.Adam(lstm_model.parameters())
loss_fn = torch.nn.BCELoss()
lstm_model.best(train_sentiment_dl, test_sentiment_dl, 10, optimizer, loss_fn, device)

