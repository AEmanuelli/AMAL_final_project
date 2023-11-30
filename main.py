import torch
import torch.nn as nn
import torch.optim as optim
from model import lmu_rnn_conv1d
from dataloader import PermutedMNIST
from torch.utils.data import DataLoader

# Paramètres pour l'initialisation de lmu_rnn_conv1d
embed_dims = 256  # Exemple de dimension d'embedding
num_heads = 8  # Exemple de nombre de têtes d'attention
mlp_ratio = 4.  # Exemple de ratio pour les couches MLP
depth = 6  # Exemple de profondeur de chaque bloc
num_classes = 10  # Nombre de classes dans psMNIST
img_size = 28  # Taille de l'image dans psMNIST (28x28)

# Paramètres d'entraînement
num_epochs = 50
learning_rate = 0.0001
batch_size = 100
lr_decay_factor = 0.85
lr_decay_step = 5

# Création des datasets et DataLoaders
train_dataset = PermutedMNIST(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = PermutedMNIST(train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Création du modèle
model = lmu_rnn_conv1d(pretrained = False, num_classes = 10)  # Utilisation du modèle défini dans model.py

# Critère de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Mise à jour du taux d'apprentissage
    if (epoch + 1) % lr_decay_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_factor

    # Affichage des informations d'entraînement
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")