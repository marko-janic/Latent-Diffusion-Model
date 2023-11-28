import numpy as np
import torch


def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen, :sidelen], axis=-1)[None, ...].astype(np.float32)
    pixel_coords /= sidelen
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords


def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
      device: Device for torch

    Returns:
      The standard deviation.
    """
    t.to(device)
    #t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
      device: Device for torch

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=device)


def compute_snr(x, x_hat):
    """Returns SNR of x_hat wrt to gt image x."""

    if x.shape != x_hat.shape:
        raise Exception("x and x_hat don't have the same dimensions: " + str(x.shape) + " vs " + str(x_hat.shape))

    diff = x - x_hat
    return -20 * np.log10(np.linalg.norm(diff) / np.linalg.norm(x))
