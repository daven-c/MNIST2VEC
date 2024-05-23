import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import matplotlib.cm as cm
from models import EmbeddingModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = EmbeddingModel().to(device)
model.load_state_dict(torch.load('./modelsaves/embedding_model.pth'))
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, shuffle=False)

with torch.no_grad():
    test_embeddings = []
    test_labels = []
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_embeddings.append(outputs.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

test_embeddings = np.concatenate(test_embeddings, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

x = test_embeddings[:, 0]
y = test_embeddings[:, 1]
z = test_embeddings[:, 2]

# Choose a colormap (e.g., 'viridis', 'plasma')
cmap = cm.get_cmap('plasma', 10)
colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
          for c in cmap(np.arange(10))]

# Create the visualization plot
fig = go.Figure(
    go.Scatter3d(x=x,
                 y=y,
                 z=z,
                 mode='markers',
                 marker=dict(
                     size=3,
                     color=[colors[label] for label in test_labels],
                     opacity=1,
                     line=dict(width=0)
                 ),
                 hovertext=test_labels,
                 hoverinfo='text',
                 ),
)

fig.update_layout(template='plotly_dark',
                  paper_bgcolor='black', plot_bgcolor='black')
fig.show()
