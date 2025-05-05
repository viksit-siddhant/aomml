import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchtext
from torch import nn
from optim import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from imdb import train_dataloader, vocab

def moving_average(data, k):
    data = np.array(data)
    return np.convolve(data, np.ones(k)/k, mode='valid')

# Define a simple CNN for MNIST
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(14*14*32, 10)
)

loss_fn = nn.CrossEntropyLoss()

def loop(dataloader, model, loss_fn, optimizer, lr = 0.001,epochs=1, averaging = False):
    optim = optimizer(model.parameters(), lr=lr)
    if averaging:
        avg = Averager(model)
    loss_curve = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for X,y in tqdm(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            if averaging:
                avg.update(model)
                model = avg.get_model()
            loss_curve.append(loss.item())
    return loss_curve

def two_loop_optimizer(dataloader, model, loss_fn, optimizer1, optimizer2, lr = 0.001, epochs=1, threshold=0.1):
    optim1 = optimizer1(model.parameters(), lr=lr)
    optim2 = optimizer2(model.parameters(), lr=lr)
    loss_curve = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i,(X,y) in tqdm(enumerate(dataloader)):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            if i <= threshold*len(dataloader):
                # Backpropagation
                optim1.zero_grad()
                loss.backward()
                optim1.step()
            else:
                optim2.zero_grad()
                loss.backward()
                optim2.step()
            loss_curve.append(loss.item())
    return loss_curve


def zeroth_order_loop(dataloader, model, loss_fn, optimizer, lr = 0.001, epochs=1):
    optim = optimizer(model.parameters(), lr=lr, loss_fn=loss_fn, rand_directions=2)
    loss_curve = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for X,y in tqdm(dataloader):
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
            optim.step(model, X, y)
            #print(f"Loss: {loss.item()}")
            loss_curve.append(loss.item())
    return loss_curve

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
optimizers = [CustomOptimizer]
plt.figure(figsize=(10, 6))
for i,optimizer in enumerate(optimizers):
    print(f"Training with {optimizer.__name__}")
    if optimizer == CustomZeroOrderOptimizer:
        loss_curve = zeroth_order_loop(train_loader, model, loss_fn, optimizer)
    else:
    #    loss_curve = loop(train_loader, model, loss_fn, optimizer,averaging=True)
        continue
  #  plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
plt.title('Loss Curve')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')

rnn = nn.Sequential(
    nn.Embedding(len(vocab), 50, padding_idx=vocab['<pad>']),
    nn.RNN(50, 128, batch_first=True),
    nn.Linear(128, 2)
)
loss_fn = nn.CrossEntropyLoss()
loss_curve = loop(train_dataloader, rnn, loss_fn, CustomOptimizer, lr=0.01, epochs=1)
plt.plot(moving_average(loss_curve,100), label='RNN')
plt.savefig('loss_curve_rnn.png')