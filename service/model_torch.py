import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    RNN LSTM model to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
    
    
    def fit(self, train_loader, epochs, optimizer, loss_fn, device):
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0
            for batch in train_loader:         
                batch_X, batch_y = batch            
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)            
                optimizer.zero_grad()
                out = self.forward(batch_X)
                loss = loss_fn(out, batch_y)
                loss.backward()
                optimizer.step()            
                total_loss += loss.data.item()
            print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
