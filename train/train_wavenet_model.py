# # training wavenet model

import numpy as np
import torch
import torch.nn.functional as F

with open('../Dataset/names.txt','r') as f:
    words = f.read().split('\n')

stoi = {chr(i):(i-96) for i in range(97,122+1)}
stoi['.']=0
itos = {stoi[k]:k for k in stoi}

def gen_data(words,block_size):    
    X = []
    Y = []
    for word in words:
        features = [0] * block_size   # [[0],[0],[0]]
        for ch in word:
            index = stoi[ch]        # target for features
            X.append(features)
            Y.append(index)
            # if word ends
            if index == 0:
                break
            features = features[1:] + [index]
        X.append(features)
        Y.append(0)
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X, Y

g = torch.Generator()
g.manual_seed(42)
num_embedding = 24
num_tanh = 128
block_size = 8
vocab_size = 27

# 80% training, 10% cv, 10% test
import random
random.seed(42)
random.shuffle(words)

n1=int(0.8*len(words))
n2=int(0.9*len(words))

Xtr, Ytr = gen_data(words[:n1],block_size)
Xcv, Ycv = gen_data(words[n1:n2],block_size)
Xte, Yte = gen_data(words[n2:],block_size)

class Linear():
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight = torch.randn((fan_in,fan_out),generator = g) / (fan_in**0.5)
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
          self.out += self.bias
        return self.out
  
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

class Embedding:
    
    def __init__(self,vocab_size,num_embedding):
        self.weight = torch.randn((vocab_size,num_embedding),generator=g)
    
    def __call__(self,x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]
    
class Flatten:

    def __init__(self,n):
        self.n = n

    def __call__(self,x):
        if x.shape[1] == self.n:
            # (..,1,..) - we will squeeze this layer
            self.out = x.view(x.shape[0],-1)
        else:
            self.out = x.view(x.shape[0],x.shape[1]//self.n,-1)
        return self.out

    def parameters(self):
        return []

class BatchNorm1d:
  
  def __init__(self, num_of_features, eps=1e-5, momentum=0.05):
    self.eps = eps
    self.momentum = momentum
    self.training = True
      
    # parameters (trained with backprop)
    self.gamma = torch.ones(num_of_features)
    self.beta = torch.zeros(num_of_features)
      
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(num_of_features)
    self.running_var = torch.ones(num_of_features)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
        if x.ndim == 3:
            dim = (0,1)
        elif x.ndim == 2:
            dim= 0
        xmean = x.mean(dim, keepdim=True) # batch mean
        xvar = x.var(dim, keepdim=True) # batch variance
        
    else:
      xmean = self.running_mean
      xvar = self.running_var
        
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
      
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    
    return self.out
  
  def parameters(self):
    # updated using backprop
    return [self.gamma, self.beta]

class Sequential:

    def __init__(self,layers):
        self.layers = layers

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        
        return self.out

    def parameters(self):
        # double list comprehesion
        return [p for layer in self.layers for p in layer.parameters()]


model = Sequential([
    Embedding(vocab_size,num_embedding),
    Flatten(2),Linear(2*num_embedding,num_tanh,bias=False), BatchNorm1d(num_tanh), Tanh(),
    Flatten(2),Linear(2*num_tanh,num_tanh,bias=False), BatchNorm1d(num_tanh), Tanh(),
    Flatten(2),Linear(2*num_tanh,num_tanh,bias=False), BatchNorm1d(num_tanh), Tanh(),
    Linear(num_tanh,vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1           # making softmax less-confident initially

params = model.parameters()

num_params=0
for p in params:
    p.requires_grad = True
    num_params += p.nelement()

@torch.no_grad()
def get_loss(X,Y):
    logits = model(X)
    loss = F.cross_entropy(logits,Y)
    return loss.item()

print("Initial Loss:",get_loss(Xtr,Ytr))

loss_hist=[]

for epoch in range(25):
    
    #mini-batching
    batch_size = 40
    idx = np.arange((Ytr.shape[0]))
    num_iter = int(Ytr.shape[0]/batch_size)
    
    for j in range(num_iter):

        # batch
        Xb = Xtr[idx[j*batch_size:(j+1)*batch_size]]
        Yb = Ytr[idx[j*batch_size:(j+1)*batch_size]]

        # forward pass
        logits = model(Xb)
        loss = F.cross_entropy(logits,Yb)
        loss_hist.append(loss.item())

        # backward pass
        for p in params:
            p.grad=None
        loss.backward()

        # gradient descent
        alpha = 0.1 if epoch<10 else 0.05 if epoch<15 else 0.01 if epoch<20 else 0.001
        for p in params:
            p.data -= alpha * p.grad

    print(f"Epoch :{epoch+1}  Loss:{get_loss(Xtr,Ytr)}")

# put layers into eval mode (needed for batchnorm especially)
for layer in model.layers:
  layer.training = False

print("Training Loss: ",get_loss(Xtr,Ytr))
print("Test Loss", get_loss(Xte,Yte))

# generating samples
# sample from the model

with open('../samples_wavenet.txt','a') as f:

    for _ in range(100):
        
        out = []
        context = [0] * (block_size)
        while True:
            # forward pass the neural net
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            # shift the context window and track the samples
            context = context[1:] + [ix]
            out.append(ix)
            # if we sample the special '.' token, break
            if ix == 0:
                break
        
        temp = ''.join(itos[i] for i in out)
        f.write(temp+'\n')