
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.ToTensor()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Train_dataset = datasets.MNIST(
    root="./MNIST",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=Train_dataset,
    batch_size=16,
    shuffle=True
)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder Blocks
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30)  # Bottleneck
        )

        # Decoder Blocks
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
        return reconstructed


def train_main():
    model = Autoencoder().to(device=DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    EPOCHS = 10

    # Saving model for later use
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, _ in train_loader:
            images = images.view(images.size(
                0), -1).to(DEVICE)  # Flatten Images
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "BestAutoencoderModel.pth")

    print("Complete Training")


if __name__ == "__main__":
    train_main()
