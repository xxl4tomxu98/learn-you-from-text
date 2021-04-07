import torch.nn as nn
import torch

class Sentiment(nn.Module):
    """
    RNN LSTM model to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(Sentiment, self).__init__()
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
        texts = x[1:,:]
        embeds = self.embedding(texts)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
    
    
    def fit(self, train_loader, optimizer, loss_fn, device):        
        # initialize epoch
        total_loss, binary_acc = 0, 0
        self.train()            
        for batch in train_loader:         
            batch_X, batch_y = batch            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # reset gradients every new batch            
            optimizer.zero_grad()
            # compute model output
            out = self.forward(batch_X)
            # compute loss and binary acc
            loss = loss_fn(out, batch_y)
            acc = self.binary_accuracy(out, batch_y)
            # back propagate loss and compute gradients
            loss.backward()
            # update weights, loss and acc
            optimizer.step()            
            total_loss += loss.data.item()
            binary_acc += acc.item()
        return total_loss / len(train_loader), binary_acc / len(train_loader)
    

    # define metric
    def binary_accuracy(self, preds, y):
        #round predictions to the closest integer
        rounded_preds = torch.round(preds)        
        correct = (rounded_preds == y).float() 
        acc = correct.sum() / len(correct)
        return acc

    
    def evaluate(self, test_loader, loss_fn, device):    
        #initialize every epoch        
        total_loss, binary_acc = 0, 0
        self.eval()        
        #deactivates autograd
        with torch.no_grad():            
            for batch in test_loader:                
                batch_X, batch_y = batch            
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)                
                # eval the model using test set
                out = self.forward(batch_X)                
                #compute valid loss and accuracy
                loss = loss_fn(out, batch_y)
                acc = self.binary_accuracy(out, batch_y)                
                #keep track of loss and accuracy
                total_loss += loss.data.item()
                binary_acc += acc.item()            
        return total_loss / len(test_loader), binary_acc / len(test_loader)


    def best(self, train_loader, test_loader, epochs, optimizer, loss_fn, device):
        best_valid_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.fit(train_loader, optimizer, loss_fn, device)
            valid_loss, valid_acc = self.evaluate(test_loader, loss_fn, device)
            #save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'static/pytorch/saved_weights.pt')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')