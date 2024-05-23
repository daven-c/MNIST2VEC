import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import EmbeddingModel, EmbeddingDecoder


def load_configs(path: str = "config.cfg"):
    with open(path, "r") as f:
        lines = f.readlines()
        settings = {}
        for line in lines:
            data = line.rstrip().split("=")
            if len(data) == 2:
                if data[1].count(".") > 0:  # Floats
                    data[1] = float(data[1])
                elif data[1].isdigit():  # Ints
                    data[1] = int(data[1])
                elif data[1] == "True":  # Bool
                    data[1] = True
                elif data[1] == "False":
                    data[1] = False
                settings.update({data[0]: data[1]})
        return settings


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    configs = load_configs()

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Create models
    embedding_model = EmbeddingModel().to(device)
    decoder_model = EmbeddingDecoder().to(device)

    # Single optimizer for both enc/dec
    optimizer = torch.optim.Adam(
        list(embedding_model.parameters()) + list(decoder_model.parameters()), lr=configs["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train the models
    epochs = configs["epochs"]
    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through both models
            embeddings = embedding_model(inputs)
            decoded_outputs = decoder_model(embeddings)

            # Calculate loss and propogate
            loss = criterion(decoded_outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate performance
            predicted = torch.argmax(decoded_outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(inputs) * 100

            # Print status every few batches (default 100)
            if (i + 1) % configs["batch_print_interval"] == 0:
                print(
                    f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')

    # Test the model
    embedding_model.eval()
    decoder_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            embeddings = embedding_model(images)
            outputs = decoder_model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Overall Accuracy: {100 * correct / total:.2f}%')

    # Save model weights
    torch.save(embedding_model.state_dict(),
               "./modelsaves/embedding_model.pth")
    torch.save(decoder_model.state_dict(), "./modelsaves/decoder_model.pth")
