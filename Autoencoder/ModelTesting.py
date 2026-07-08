
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from Autoencoder import Autoencoder
import matplotlib.pyplot as plt
transform = transforms.ToTensor()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder().to(DEVICE)
model.load_state_dict(torch.load(
    "BestAutoencoderModel.pth", map_location=DEVICE))
model.eval()


test_dataset = datasets.MNIST(
    root="./MNIST_Test",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=True
)

images, _ = next(iter(test_loader))
images = images.view(images.size(0), -1).to(DEVICE)

with torch.no_grad():
    reconstructed = model(images)

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
