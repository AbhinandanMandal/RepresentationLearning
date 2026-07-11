
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.ToTensor()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.MNIST(
    root="./",
    train=True,
    transform=transform,
    download=True)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True
)


class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

        # Encoder block
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30)  # Bottleneck
        )

        # Decoder Block
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent  # latent is needed for calculating KL divergence


def KLDivergence(latent):
    rho = 0.05
    eps = 1e-8
    rho_hat = latent.mean(dim=0)
    # To fix problem of vanishing/exploding gradient
    rho_hat = torch.clamp(rho_hat, eps, 1-eps)
    KL = rho * torch.log(rho / rho_hat) + (1 - rho) * \
        torch.log((1-rho)/(1-rho_hat))
    return KL.sum()


def train_main():
    model = SparseAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    beta = 0.0001
    EPOCHS = 10
    best_loss = float("inf")  # For saving best model during training

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(DEVICE)
            outputs, latent = model(images)
            reconstrcuted_loss = criterion(outputs, images)
            sparsity_loss = KLDivergence(latent)
            loss = reconstrcuted_loss + beta*sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "BestSparseAutoencoderModel.pth")
    print("---------------Training Completed-----------------")


if __name__ == "__main__":
    train_main()
