import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
from skimage.transform import radon, iradon


# dataset_path = os.path.join("..", "datasets", "limited-CT_64", "limited-CT", "horizontal_snr25.0.npz")  # on sciCORE
dataset_path = "C:/Users/marko/Desktop/Bachelors Thesis/datasets/limited_CT/" \
               "limited-CT_64/limited-CT/horizontal_snr25.0.npz"  # on local machine

n_rotations_per_image = 5
n_reconstruction_angles = 90  # number of angles used for linear space for fbp
theta_low = 30  # lower value for angles
theta_max = 120  # higher value for angles
angles = np.linspace(theta_low, theta_max, n_reconstruction_angles)


def get_rotation_matrix(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def main():
    data = np.load(dataset_path)
    images = torch.from_numpy(data["x_train"]).float()

    ground_truth = images[0, :, :, 0]
    ground_truth = ground_truth[np.newaxis, :, :]
    image_fbp = iradon(radon(ground_truth[0].detach().cpu().numpy(), angles, circle=False), angles, circle=False)
    image_fbp = torch.from_numpy(image_fbp[np.newaxis, :, :])
    print(image_fbp.shape)

    rotator = v2.RandomRotation(degrees=(0, 360), fill=True)
    rotated_images = [rotator.forward(image_fbp) for _ in range(n_rotations_per_image)]

    plt.imshow(ground_truth[0], cmap="gray")
    plt.show()
    for image in rotated_images:
        plt.imshow(image[0], cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
