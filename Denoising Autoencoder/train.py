
from pathlib import Path

# from data import get_dataloader
# from utils import add_gaussian_noise
# from model import DenoisingAutoencoder
try:
    from .data import get_dataloader
    from .utils import add_gaussian_noise
    from .model import DenoisingAutoencoder
except ImportError:  # Supports direct execution: python train.py
    from data import get_dataloader
    from utils import add_gaussian_noise
    from model import DenoisingAutoencoder
import torch.nn as nn
import torch
import torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# train_loader, test_loader = get_dataloader(batch_size=16)
MODEL_PATH = Path(__file__).resolve().parent / "DenoisingAutoencoder_best.pt"


def train_main(batch_size: int = 16, epochs: int = 10):
    """Train the model to reconstruct clean images from noisy inputs."""
    train_loader, _ = get_dataloader(batch_size=batch_size)
    model = DenoisingAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # EPOCHS = 10
    EPOCHS = epochs
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(DEVICE)
            noisy_images = add_gaussian_noise(images)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # torch.save(model.state_dict(), "DenoisingAutoencoder_best.pt")
            torch.save(model.state_dict(), MODEL_PATH)
    print("---------------------Complete Training--------------------------")


if __name__ == "__main__":
    train_main()
