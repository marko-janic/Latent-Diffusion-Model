# Library imports
import torch
import numpy as np
from omegaconf import OmegaConf

# Local file imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler


# Dataset
dataset_path = ("/mnt/c/Users/marko/Desktop/Bachelors Thesis/datasets/limited_CT/"
                "limited-CT_64/limited-CT/horizontal_snr25.0.npz")  # local machine wsl

# Autoencoder Model
config_path = "models/vq-f4/config.yaml"
ckpt_path = "models/vq-f4/model.ckpt"

# Pytorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Torch Device: " + str(device))


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    data = np.load(dataset_path)
    images = torch.from_numpy(data["x_train"]).float()
    x = images[np.newaxis, 0:3, :, :, 0]
    x = x.to(device)
    print("Shape of x: " + str(x.shape))

    # Load pretrained autoencoder
    config = OmegaConf.load(config_path)
    autoencoder_model = load_model_from_config(config, ckpt_path)
    autoencoder_model = autoencoder_model.to(device)

    x_encoded = autoencoder_model.encode(x)


if __name__ == "__main__":
    main()
