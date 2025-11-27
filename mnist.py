import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

MODEL_PATH = "/fs/dev/neural-nets/model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

# 2. Load the Training Data
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 3. Load the Test Data
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class SimpleNetwork(nn.Module):

  def __init__(self):
    super().__init__()
    
    self.layer1 = nn.Linear(28*28, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.layer1(x))
    x = self.layer2(x)
    x = self.layer3(x)

    return x


network = SimpleNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)

if os.path.exists(MODEL_PATH):
    print("Loading previously trained model from saved path")
    network.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("No previously trained model found, training now")
    epochs = 20
    for e in range(epochs):
      network.train()
      loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        prediction = network(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
      
      print(f"Loss for epoch {e}: {loss}")

    torch.save(network.state_dict(), MODEL_PATH)

network.eval()
correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
    scores = network(images)
    _, result = scores.max(1)

    correct += (result == labels).sum()
    total += result.size(0)

  print(f"Correct = {correct}, Total = {total}, Accuracy = {(correct/total)*100}")
