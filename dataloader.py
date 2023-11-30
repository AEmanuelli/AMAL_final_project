import torch
from torch.utils.data import Dataset, DataLoader
from datamaestro import prepare_dataset
import numpy as np
import matplotlib.pyplot as plt

class PermutedMNIST(Dataset):
    def __init__(self, train=True):
        # Chargement des données MNIST via datamaestro
        ds = prepare_dataset("com.lecun.mnist")
        self.images, self.labels = (ds.train.images.data(), ds.train.labels.data()) if train else (ds.test.images.data(), ds.test.labels.data())

        # Création d'une permutation fixe
        self.permutation = np.random.permutation(28 * 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, orig = False):
        # Application de la permutation
        original_image = self.images[idx].reshape(1, 28, 28)
        image = self.images[idx].reshape(-1)[self.permutation].reshape(1, 28, 28)
        label = self.labels[idx]
        if orig : 
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.int64), torch.tensor(original_image, dtype=torch.int64)
        else : 
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


# Chargement d'une image échantillon et de son label depuis le DataLoader
# Affichage de l'image originale et de l'image permutée
# for i, (permuted_images, labels, original_image) in enumerate(test_loader): #on regrade que ça marche bien sur le test, sur le train c random sinon
    
#     original_image = original_image[4]
#     permuted_image = permuted_images[4]
#     label = labels[4].item()

#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_image[0], cmap='gray')
#     plt.title(f'Original Image, Label: {label}')

#     plt.subplot(1, 2, 2)
#     plt.imshow(permuted_image[0], cmap='gray')
#     plt.title(f'Permuted Image, Label: {label}')
#     plt.show()

#     break