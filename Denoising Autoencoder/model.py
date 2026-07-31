
import torch
import torch.nn as nn 

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

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
        if x.ndim != 2 or x.shape[1] != 784:
            raise ValueError("Expected x with shape (batch_size, 784)")
        latent = self.encoder(x) # Inside latent, noisy image will go into
        reconstructed = self.decoder(latent)
        return reconstructed
    


