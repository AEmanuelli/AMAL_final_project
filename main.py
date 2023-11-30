from model import *
from dataloader import *


# Création des datasets et DataLoaders
train_dataset = PermutedMNIST(train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = PermutedMNIST(train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
