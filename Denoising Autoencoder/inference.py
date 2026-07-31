

from data import get_dataloader
from utils import add_gaussian_noise
from model import DenoisingAutoencoder
import torch
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = DenoisingAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(
    "DenoisingAutoencoder_best.pt", map_location=DEVICE))
model.eval()

train_dataloader, test_dataloader = get_dataloader()
images, _ = next(iter(test_dataloader))
images = images.view(images.size(0), -1).to(DEVICE)
noisy_images = add_gaussian_noise(images)

with torch.no_grad():
    reconstructed = model(noisy_images)

original = images.cpu().view(-1, 28, 28)
reconstructed = reconstructed.cpu().view(-1, 28, 28)
fig, axes = plt.subplots(2, 6, figsize=(12, 4))

for i in range(6):

    # Original
    axes[0, i].imshow(original[i], cmap='gray')
    axes[0, i].axis('off')

    # Reconstructed
    axes[1, i].imshow(reconstructed[i], cmap='gray')
    axes[1, i].axis('off')

plt.show()
