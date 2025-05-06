import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from optim import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from imdb import train_dataloader, vocab_size, PAD_IDX 

def moving_average(data, k):
    data = np.array(data)
    return np.convolve(data, np.ones(k)/k, mode='valid')

# Define a simple CNN for MNIST

loss_fn = nn.CrossEntropyLoss()

def loop(dataloader,valid_loader, model, loss_fn, optimizer, lr = 0.001,epochs=1, averaging = False):
    optim = optimizer(model.parameters(), lr=lr)
    if averaging:
        avg = Averager(model)
    loss_curve = []
    valid_curve = []
    ema_loss = [10]
    for epoch in range(epochs):
        count = 0
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
            ema_loss.append(ema_loss[-1]*0.99 + loss.item()*0.01)
            count += 1
            with torch.no_grad():
                if count % 200 == 0:
                    acc = 0
                    for X_val,y_val in valid_loader:
                        #validation accuracy
                        pred_val = model(X_val)
                        pred_val = pred_val.argmax(dim=1)
                        
                        acc += (pred_val == y_val).float().mean()
                    acc /= len(valid_loader)
                    valid_curve.append(acc.item())
            if len(ema_loss) > 400 and ema_loss[-401]-ema_loss[-1] < 0.001:
                print("Early stopping")
                break
    return loss_curve, valid_curve

def two_loop_optimizer(dataloader, valid_loader, model, loss_fn, optimizer1, optimizer2, lr = 0.001, epochs=1, threshold=0.1):
    optim1 = optimizer1(model.parameters(), lr=lr)
    optim2 = optimizer2(model.parameters(), lr=lr)
    loss_curve = []
    ema_loss = [100]
    valid_curve = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        count = 0
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
            ema_loss.append(ema_loss[-1]*0.99 + loss.item()*0.01)
            count += 1
            if len(ema_loss) > 400 and ema_loss[-401]-ema_loss[-1] < 0.001:
                print("Early stopping")
                break
            with torch.no_grad():
                if count % 200 == 0:
                    acc = 0
                    for X_val,y_val in valid_loader:
                        #validation accuracy
                        pred_val = model(X_val)
                        pred_val = pred_val.argmax(dim=1)
                        
                        acc += (pred_val == y_val).float().mean()
                    acc /= len(valid_loader)
                    valid_curve.append(acc.item())
    return loss_curve, valid_curve


def zeroth_order_loop(dataloader, valid_loader, model, loss_fn, optimizer, lr = 0.001, epochs=1, averaging = False):
    optim = optimizer(model.parameters(), lr=lr, loss_fn=loss_fn, rand_directions=2)
    loss_curve = []
    valid_curve = []
    ema_loss = [100]
    if averaging:
        avg = Averager(model)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        count = 0
        for X,y in tqdm(dataloader):
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
            optim.step(model, X, y)
            #print(f"Loss: {loss.item()}")
            loss_curve.append(loss.item())
            if averaging:
                avg.update(model)
                model = avg.get_model()
            count += 1
            ema_loss.append(ema_loss[-1]*0.99 + loss.item()*0.01)
            if len(ema_loss) > 400 and ema_loss[-401]-ema_loss[-1] < 0.001:
                print("Early stopping")
                break
            with torch.no_grad():
                if count % 200 == 0:
                    acc = 0
                    for X_val,y_val in valid_loader:
                        #validation accuracy
                        pred_val = model(X_val)
                        pred_val = pred_val.argmax(dim=1)
                        
                        acc += (pred_val == y_val).float().mean()
                    acc /= len(valid_loader)
                    valid_curve.append(acc.item())
    return loss_curve, valid_curve

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#split the dataset into train and validation sets
train_size = int(0.99 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#optimizers = [CustomOptimizer, AdagradOptimizer, AdamOptimizer, PAdagradOptimizer, [AdamOptimizer, CustomOptimizer]]
optimizers = [SGDMomentum]
for i,optimizer in enumerate(optimizers):
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(14*14*32, 10)
    )
    print(f"Training with {optimizer}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    if optimizer == CustomZeroOrderOptimizer:
        loss_curve, valid_curve = zeroth_order_loop(train_loader, valid_loader, model, loss_fn, optimizer)
        plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer.__name__}_img_noavg.png')
        continue
    try:
        catching = optimizer[0]
    except:
        pass
    else:
        loss_curve, valid_curve = two_loop_optimizer(train_loader, valid_loader, model, loss_fn, optimizer[0], optimizer[1])
        plt.plot(moving_average(loss_curve,100), label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer[0].__name__}_{optimizer[1].__name__}_img_noavg.png')
        continue
    loss_curve, valid_curve = loop(train_loader, valid_loader,model, loss_fn, optimizer)
    plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(valid_curve, label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig(f'loss_curve_{optimizer.__name__}_img_noavg.png')
    continue
for i,optimizer in enumerate(optimizers):
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(14*14*32, 10)
    )
    print(f"Training with {optimizer}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    if optimizer == CustomZeroOrderOptimizer:
        loss_curve, valid_curve = zeroth_order_loop(train_loader, valid_loader, model, loss_fn, optimizer,averaging=True)
        plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer.__name__}_img_avg.png')
        continue
    try:
        catching = optimizer[0]
    except:
        pass
    else:
        loss_curve, valid_curve = two_loop_optimizer(train_loader, valid_loader, model, loss_fn, optimizer[0], optimizer[1])
        plt.plot(moving_average(loss_curve,100), label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer[0].__name__}_{optimizer[1].__name__}_img_noavg.png')
        continue
    loss_curve, valid_curve = loop(train_loader, valid_loader, model, loss_fn, optimizer,averaging=True)

    plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(valid_curve, label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig(f'loss_curve_{optimizer.__name__}_img_avg.png')
    continue

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x
loss_fn = nn.CrossEntropyLoss()
from imdb import train_dataloader, val_dataloader
train_loader = train_dataloader
valid_loader = val_dataloader

for i,optimizer in enumerate(optimizers):
    rnn = RNN(vocab_size, 128, 128)
    print(f"Training with {optimizer}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    if optimizer == CustomZeroOrderOptimizer:
        loss_curve, valid_curve = zeroth_order_loop(train_loader, valid_loader, rnn, loss_fn, optimizer)
        plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer.__name__}_txt_noavg.png')
        continue
    try:
        catching = optimizer[0]
    except:
        pass
    else:
        loss_curve, valid_curve = two_loop_optimizer(train_loader, valid_loader, rnn, loss_fn, optimizer[0], optimizer[1])
        plt.plot(moving_average(loss_curve,100), label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer[0].__name__}_{optimizer[1].__name__}_txt_noavg.png')
        continue
    loss_curve, valid_curve = loop(train_loader, valid_loader, rnn, loss_fn, optimizer,averaging=True)
    plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(valid_curve, label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig(f'loss_curve_{optimizer.__name__}_txt_noavg.png')
    continue

for i,optimizer in enumerate(optimizers):
    print(f"Training with {optimizer}")
    rnn = RNN(vocab_size, 128, 128)
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    if optimizer == CustomZeroOrderOptimizer:
        loss_curve, valid_curve = zeroth_order_loop(train_loader, valid_loader, rnn, loss_fn, optimizer,averaging=True)
        plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer.__name__}_txt_avg.png')
        continue
    try:
        catching = optimizer[0]
    except:
        pass
    else:
        loss_curve, valid_curve = two_loop_optimizer(train_loader, valid_loader, rnn, loss_fn, optimizer[0], optimizer[1])
        plt.plot(moving_average(loss_curve,100), label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1,2,2)
        plt.plot(valid_curve, label=f'{optimizer[0].__name__}_{optimizer[1].__name__}')
        plt.xlabel('Batch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.savefig(f'loss_curve_{optimizer[0].__name__}_{optimizer[1].__name__}_txt_avg.png')
        continue
    loss_curve, valid_curve = loop(train_loader, valid_loader, rnn, loss_fn, optimizer,averaging=True)
    plt.plot(moving_average(loss_curve,100), label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(valid_curve, label=optimizer.__name__)
    plt.xlabel('Batch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig(f'loss_curve_{optimizer.__name__}_txt_avg.png')
    continue