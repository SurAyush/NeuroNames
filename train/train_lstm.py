import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

with open("../Dataset/names.txt",'r') as f:
    words = f.read().split('\n')

# modifying 
# . : end of seq
# ~ : start of seq
# padding with . to make all words of same length
max_len = 16            # 15+1
words_mod = ['~' + word.lower() + '.'*(max_len-len(word)) for word in words]

stoi = {chr(c):(c-97) for c in range(97,122+1)}
stoi['.'] = 26
stoi['~'] = 27
itos = {stoi[k]:k for k in stoi.keys()}

def encoder(l):
    ''' It will take a list of strings (words) and it will return a 2D list of each char encoded '''
    res = [list(el) for el in l]   # splitting it characterwise
    for i,el in enumerate(res):
        for j,ch in enumerate(el):
            res[i][j] = stoi[ch]
    
    return res


def decoder(l):
    ''' It will take a 2D list of int and decode it to a 1D list of strings '''
    res = []
    for el in l:
        w = ''.join(itos[ch] for ch in el)
        res.append(w)
    
    return res


data = encoder(words_mod)
n = 0.9
# shuffling data
np.random.shuffle(data)
ul = int(n*len(data))

data_train = data[:ul]
data_val = data[ul:]

# buidling the dataset
X_train = torch.tensor([el[:max_len] for el in data_train])
X_test = torch.tensor([el[:max_len] for el in data_val])
Y_train = torch.tensor([el[1:max_len + 1] for el in data_train])
Y_test = torch.tensor([el[1:max_len + 1] for el in data_val])

# hyperparameters
n_layers = 3
n_hidden = 128
n_output = 27
alpha = 1e-3
n_vocab = 28
n_emb = 128
batch_size = 640
max_len = 16
epochs = 30



class LSTM_OP(nn.Module):

    def __init__(self, n_vocab, n_emb, n_hidden, n_layers, dropout=0.1):
        super().__init__()
        self.n_emb = n_emb
        self.n_vocab = n_vocab
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_vocab, n_emb)
        self.lstm = nn.LSTM(n_emb, n_hidden, n_layers, batch_first=True, dropout=dropout)
        # batch_first=True means that the first dimension of the input tensor will be the batch size -> (batch_size, seq_len, n_features)
        self.fc = nn.Linear(n_hidden, n_emb)
        self.relu = nn.ReLU()
        self.output = nn.Linear(n_emb, n_vocab)
    
    def forward(self,x):
        '''x: (batch_size, seq_len)'''
        # pack_padded_seq can be applied to reduce computation but not necessary here
        x = self.embedding(x)                                           # (batch_size, seq_len, n_emb)
        cell_state = torch.zeros(self.n_layers, x.size(0), self.n_hidden)       # long term memory
        hidden_state = torch.zeros(self.n_layers, x.size(0), self.n_hidden)     # short term memory
        out, _ = self.lstm(x, (hidden_state, cell_state))       # we just need the long term memory
        # Since we're often interested in predicting based on the entire sequence, taking the last output represents the accumulated information over the whole sequence.
        out = self.fc(out)
        out = self.relu(out)
        out = self.output(out)

        return out
    
    def generate_words(self,X,max_len):
        
        self.eval()
        t = X.shape[1] 
            
        for i in range(max_len - t + 1):      # t:max -> 17
            out = self.forward(X)
            out = out[:,-1,:]      #(B,C)   we just need the last char
            probs = F.softmax(out,dim=-1)
            # return indices
            next_char = torch.multinomial(probs,num_samples=1)   #(B,1)
            X = torch.cat((X,next_char),dim=1)

        return X
    

    
model = LSTM_OP(n_vocab, n_emb, n_hidden, n_layers, dropout=0.2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=alpha)
model.train()

@torch.no_grad()
def get_loss():
    '''Returns the loss on the validation set'''
    out = model(X_test)
    loss = criterion(out.view(-1,n_vocab),Y_test.view(-1))
    return loss.item()

J_hist = []

# training
for epoch in range(epochs):
    for i in range(X_train.shape[0]//batch_size):
        X = X_train[i*batch_size:(i+1)*batch_size]
        y = Y_train[i*batch_size:(i+1)*batch_size]             # (batch_size, seq_len)
        model.train()
        optimizer.zero_grad()
        out = model(X)         # (batch_size, seq_len, n_vocab)
        # flatten the output and target
        out = out.view(-1, n_vocab)          # (batch_size*seq_len, n_vocab)
        y = y.view(-1)                       # (batch_size*seq_len)
        loss = criterion(out, y)
        J_hist.append(loss.item())
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            print(f'Epoch: {epoch+1} - Iteration: {i+1} Loss: {loss.item()}')
        
    print(f'Epoch: {epoch+1} Validation Loss: {get_loss()}')


# saving the model
torch.save(model.state_dict(),'lstm_model.pth')