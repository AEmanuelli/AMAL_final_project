import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import *
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

# from utils.mnist_task import PMNISTData
# def get_data_iterator(batch_size, seed):
#     data_iterator = PMNISTData(batch_size=batch_size, seed=seed)
#     return data_iterator


# Création des datasets et DataLoaders
train_dataset = PermutedMNIST(train=True)
test_dataset = PermutedMNIST(train=False)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train_loader = get_data_iterator(100, 1)
# test_loader = get_data_iterator(100, 1)

# Création du modèle
model = LMU(dim=8, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0., sr_ratio=1, use_all_h=True)  # Utilisation du modèle défini dans model.py

# Critère de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in tqdm(range(num_epochs)):
    model.train()  # Mode entraînement
    total_loss = 0
    for images, labels in tqdm(train_loader):
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