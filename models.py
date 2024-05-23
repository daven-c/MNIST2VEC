import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):

    def __init__(self, embedding_dim: int = 128):
        super(EmbeddingModel, self).__init__()

        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 64, embedding_dim)
        self.relu3 = nn.ReLU()

        self.embedding = nn.Linear(embedding_dim, 3)  # Output embedding

    def forward(self, img):
        x = self.conv1(img)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, 7 * 7 * 64)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.embedding(x)  # Generate embedding

        return x

    def predict(self, x):
        x = self.forward(x).detach()
        return x


class EmbeddingDecoder(nn.Module):

    def __init__(self):
        super(EmbeddingDecoder, self).__init__()

        # Decode embedding
        self.fc1 = nn.Linear(3, 7 * 7 * 64)
        self.relu1 = nn.ReLU()

        # Predict the MNIST label
        self.fc2 = nn.Linear(7 * 7 * 64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emb):
        x = self.fc1(emb)
        x = self.relu1(x)

        # Flatten
        x = x.view(-1, 7 * 7 * 64)

        x = self.fc2(x)
        x = self.softmax(x)

        return x
