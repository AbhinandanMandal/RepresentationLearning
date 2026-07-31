
# Main idea of Denoising Autoencoder is to pass an image with noise into it
# Right now, we're using Gaussian Noise for primary testing

import torch


def add_gaussian_noise(images: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    """Add zero-mean Gaussian noise and retain valid normalized pixel values."""
    if std < 0:
        raise ValueError("std must be non-negative")

    # noise = torch.rand_like(images)*std
    noise = torch.randn_like(images) * std
    # noisy_image = images + noise
    noisy_images = images + noise
    # noisy_image = torch.clamp(noisy_image, 0., 1.) # torch.clamp fits all tensor within specified min and max value
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    # return noisy_image
    return noisy_images



    
