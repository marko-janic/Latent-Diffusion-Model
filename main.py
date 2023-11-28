import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_path = "C:/Users/marko/Desktop/Bachelors Thesis/datasets/limited_CT/" \
               "limited-CT_64/limited-CT/horizontal_snr25.0.npz"  # on local machine


def main():
    data = np.load(dataset_path)
    images = torch.from_numpy(data["x_train"]).float()
    x = images[0, :, :, 0]

    # Load pretrained autoencoder
    autoencoder_model = torch.load("vq-f4/model.ckpt", map_location=device)
    autoencoder_model = autoencoder_model.to(device)

    x_encoded = autoencoder_model.decode_first_stage(x)
    plt.imshow(x)
    plt.show()


if __name__ == "__main__":
    main()
