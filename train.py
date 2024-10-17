import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Create a xor dataset with some random noise
class XORDataset(Dataset):
    def __init__(self, numSamples=100, noise=0.01):
        self.len = numSamples
        self.x = torch.zeros((numSamples, 2))
        self.y = torch.zeros((numSamples, 1))
        for i00 in range(numSamples // 4):
            i01 = i00 + numSamples // 4
            i10 = i00 + numSamples // 2
            i11 = i00 + 3 * numSamples // 4
            self.x[i00, :] = torch.Tensor([0.0, 0.0])
            self.y[i00, 0] = torch.Tensor([0.0])
            self.x[i01, :] = torch.Tensor([0.0, 1.0])
            self.y[i01, 0] = torch.Tensor([1.0])
            self.x[i10, :] = torch.Tensor([1.0, 0.0])
            self.y[i10, 0] = torch.Tensor([1.0])
            self.x[i11, :] = torch.Tensor([1.0, 1.0])
            self.y[i11, 0] = torch.Tensor([0.0])

            self.x = self.x + noise * torch.randn((numSamples, 2))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

# Define the class model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# Define accuracy
def accuracy(model, dataSet):
    return np.mean(dataSet.y.view(-1).numpy() == (model(dataSet.x)[:, 0] > 0.5).numpy())

model = Net()
dataSet = XORDataset()

# training
learningRate = 0.1
epochs = 500
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
train_loader = DataLoader(dataset=dataSet, batch_size=1)
for epoch in range(epochs):
    totalLoss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalLoss+=loss.item()
    print('epoch[{}/{}]: loss[{}] accuracy[{}]'.format(epoch+1, epochs, totalLoss, accuracy(model, dataSet)))

# export weight
model.linear1.weight.data.numpy().astype('float32').tofile('linear1_weight.fp32')
model.linear1.bias.data.numpy().astype('float32').tofile('linear1_bias.fp32')
model.linear2.weight.data.numpy().astype('float32').tofile('linear2_weight.fp32')
model.linear2.bias.data.numpy().astype('float32').tofile('linear2_bias.fp32')