import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Embedder(nn.Module):
    OUTPUT_DIM = 3

    def __init__(self, input_size: int, hidden_dim: int):
        super(Embedder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()

        self.embedding = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(-1, 7 * 7 * 64)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.embedding(x)

        return x


if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    print(trainloader.dataset)
    quit()

    # Create the model, optimizer, and loss function
    model = EmbeddingModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()  # Mean squared error for embedding

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.zeros_like(
                outputs))  # Zero-centered target
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}')

    # Get embeddings for testing
    with torch.no_grad():
        test_embeddings = []
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
