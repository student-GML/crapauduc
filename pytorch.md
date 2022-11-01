# Introduction à Pytorch

## Introduction

Pytorch est une bibliothèque de calcul scientifique open source basée sur le langage Python. Elle est principalement utilisée pour la recherche en apprentissage automatique et la création de réseaux de neurones. Elle est développée par Facebook et est largement utilisée dans les entreprises et les laboratoires de recherche.

## Installation

Pour installer Pytorch, vous pouvez utiliser la commande suivante:

```bash
pip install torch torchvision
```

## Création d'un modèle

Pour créer un modèle, il faut d'abord créer une classe qui hérite de la classe `nn.Module` de Pytorch. Cette classe contient les fonctions suivantes:

- `__init__`: Cette fonction est appelée lors de la création de l'objet. Elle permet d'initialiser les variables de la classe.
- `forward`: Cette fonction est appelée lors de l'appel de l'objet. Elle permet de définir le modèle.

Voici un exemple de modèle convolutif:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

## Sauvegarde des paramètres du modèle

Pytorch permet de sauvegarder les paramètres du modèle et des optimizer avec state_dict.

Voici un exemple pour sauvegarder un modèle:

```python
# Specify a path
PATH = "state_dict_model.pt"

# Save
torch.save(net.state_dict(), PATH)

# Load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
```

## Entraînement d'un modèle

Pour entraîner un modèle, il faut d'abord créer un objet `DataLoader` qui permet de charger les données. Ensuite, il faut créer un objet `Model` et un objet `Optimizer`. Enfin, il faut créer une boucle d'entraînement qui va itérer sur les données et mettre à jour les paramètres du modèle.

Voici un exemple complet:

```python

# 1. Import necessary libraries for loading our data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 2. Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. Build the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. Define a Loss function and optimizer

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 5. Zero the gradients while training the network

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

## Changement de matériel GPU CPU

Pytorch permet de changer de matériel pour l'entraînement. Par défaut, Pytorch utilise le CPU. Pour utiliser le GPU, il faut utiliser la fonction `to`:

Exemple: Save on GPU, Load on CPU

```python
# Specify a path to save to
PATH = "model.pt"

# Save
torch.save(net.state_dict(), PATH)

# Load
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))
```

Exemple: Save on CPU, Load on GPU

```python
# Save
torch.save(net.state_dict(), PATH)

# Load
device = torch.device("cuda")
model = Net()
# Choose whatever GPU device number you want
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
model.to(device)
```

## Evaluation des modèles

Pytorch offre une variété d'outils pour évaluer la performance et évaluer le modèle. Voici quelques exemples:
- [benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [visualisation](https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html)
- [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

