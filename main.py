# Library imports
import torch
import functools
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage.transform import radon, iradon
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from ema import ExponentialMovingAverage
from tqdm import tqdm

# Local file imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler
from functions import marginal_prob_std
from functions import diffusion_coeff
from models.models import ScoreNet


# Datasets
#limited_CT_dataset_path = ("/mnt/c/Users/marko/Desktop/Bachelors Thesis/datasets/limited_CT/"
#                           "limited-CT_64/limited-CT/horizontal_snr25.0.npz")  # local machine wsl
limited_CT_dataset_path = os.path.join("..", "bachelors_thesis", "datasets", "limited-CT_64",
                                       "limited-CT", "horizontal_snr25.0.npz")  # on sciCORE
image_size = 64

# Autoencoder Model
autoencoder_config_path = "models/vq-f4/config.yaml"
autoencoder_ckpt_path = "models/vq-f4/model.ckpt"
encoded_image_size = 16
encoded_image_min = -5.43
encoded_image_max = 4.57

# Pytorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Torch Device: " + str(device))

# Training
batch_size = 32
num_workers_train = 4
num_workers_test = 4
global_sigma = 25.0
skip_training = False
n_epochs = 200
learning_rate = 1e-2
losses = []
n_skip_epochs = 4  # Number of epochs to skip for plotting the training
image_channels = 3
experiment_dir = "experiments/experiment4/"  # Directory where checkpoint is

# Sampling
num_steps = 100  # Number of sampling steps
sample_batch_size = 4
n_angles = 90
theta_low = 0  # lower value for angles
theta_max = 90  # higher value for angles
angles = np.linspace(theta_low, theta_max, n_angles)
include_gradient_descent = False
n_posterior_samples = 1
global_tau = 0.5
sampling_dir = "sampling2/"  # Directory for current sampling instance


class LimitedCT64X64Loader(Dataset):
    def __init__(self, path, option, size=(64, 64)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.data = np.load(path)

        if option == "train":
            self.img_dataset = torch.from_numpy(self.data["x_train"]).float()
        elif option == "test":
            self.img_dataset = torch.from_numpy(self.data["x_test"]).float()
        else:
            raise Exception("Incorrect option for loading Dataset (Chose train or test)")

        self.im_size = size

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, index):
        img = self.img_dataset[index][:, :, 0]  # Weird array access bcs dataset image has shape (64, 64, 1)
        img = Image.fromarray(np.uint8(img), 'L')  # Transforms to PIL image for self.transform
        img = self.transform(img)
        return img


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


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t, device=x.device)

    perturbed_x = x + z * std[:, None, None, None]

    score = model(perturbed_x, random_t)

    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss


def euler_maruyama_sampler(score_model,
                           y,
                           marginal_prob_std,
                           diffusion_coeff,
                           autoencoder_model,
                           batch_size,
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3,
                           z=None,
                           tau=0.5):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      y: Numpy Array with images that are used to make function conditional, number of images should be equal to
        batch_size
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.
      tau: Used for conditional step of the calculation
    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = z

    with (torch.no_grad()):
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step, device=device)

            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size

            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

            # Add filtered back projection to make sampler conditional
            if include_gradient_descent:
                for im_index in range(batch_size):
                    x_decoded = autoencoder_model.decode(x[im_index:im_index+1])
                    x_decoded = x_decoded[0, :, :, :].cpu().detach().numpy()

                    new_x = np.zeros_like(x_decoded)
                    new_x[0] = x_decoded[0] - tau * iradon(radon(x_decoded[0], angles, circle=False) - y[im_index],
                                                           angles, circle=False)
                    new_x[1] = x_decoded[1] - tau * iradon(radon(x_decoded[1], angles, circle=False) - y[im_index],
                                                           angles, circle=False)
                    new_x[2] = x_decoded[2] - tau * iradon(radon(x_decoded[2], angles, circle=False) - y[im_index],
                                                           angles, circle=False)

                    new_x = torch.from_numpy(new_x[np.newaxis, :, :, :]).to(device)
                    x_encoded = autoencoder_model.encode_to_prequant(new_x)

                    x[im_index, :, :, :] = x_encoded[0]  # convert back to pytorch tensor
    return mean_x


def generate_samples(marginal_prob_std_fn, score_model, diffusion_coeff_fn, im_dataset, autoencoder_model):
    global device
    t = torch.ones(sample_batch_size, device=device)

    z_init = (torch.randn(n_posterior_samples, sample_batch_size, 3, encoded_image_size, encoded_image_size,
                          device=device) * marginal_prob_std_fn(t, device=device)[:, None, None, None])

    # Load the pre-trained checkpoint from disk.
    ckpt = torch.load(experiment_dir + 'ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    # Get true images
    true_images = []
    for i in range(sample_batch_size):
        true_images.append(im_dataset.__getitem__(i).cpu().detach().numpy()[0])

    # Get measurements from training set to make sampling conditional
    y = []
    for i in range(sample_batch_size):
        y.append(radon(true_images[i], angles, circle=False))  # im from dataset that makes it conditional

    # Compute FBP reconstruction
    samples_fbp_clean = []
    for i in range(sample_batch_size):
        samples_fbp_clean.append(iradon(y[i], angles, circle=False))

    samples_clean = []
    for i in range(n_posterior_samples):
        posterior_samples_clean = euler_maruyama_sampler(score_model,
                                                         y,
                                                         marginal_prob_std_fn,
                                                         diffusion_coeff_fn,
                                                         batch_size=sample_batch_size,
                                                         autoencoder_model=autoencoder_model,
                                                         device=device,
                                                         z=z_init[i], tau=global_tau)
        # posterior_samples_clean = posterior_samples_clean.clamp(0.0, 1.0)
        samples_clean.append(posterior_samples_clean)
    plt.imshow(samples_clean[0][0].permute(1, 2, 0).cpu().detach().numpy())
    plt.savefig(experiment_dir + sampling_dir + "encoded_sample.jpg")
    plt.imshow(autoencoder_model.decode(samples_clean[0])[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.savefig(experiment_dir + sampling_dir + "sample.jpg")


def main():
    # Load dataset
    train_dataset = LimitedCT64X64Loader(path=limited_CT_dataset_path,
                                         option="train",
                                         size=(image_size, image_size))
    test_dataset = LimitedCT64X64Loader(path=limited_CT_dataset_path,
                                        option="test",
                                        size=(image_size, image_size))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers_train,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers_train,
                                              shuffle=True)

    # Load Model
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=global_sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=global_sigma)

    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, image_channels=image_channels)
    score_model = score_model.to(device)

    num_param = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    print('---> Number of trainable parameters of Autoencoder: {}'.format(num_param))

    # Load Autoencoder Model
    autoencoder_config = OmegaConf.load(autoencoder_config_path)
    autoencoder_model = load_model_from_config(autoencoder_config, autoencoder_ckpt_path)
    autoencoder_model = autoencoder_model.to(device)

    if not skip_training:
        optimizer = Adam(score_model.parameters(), lr=learning_rate)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        pbar = tqdm(range(n_epochs))

        for epoch in pbar:
            avg_loss = 0.
            num_items = 0

            for x in train_loader:
                x = x.repeat(1, 3, 1, 1)  # Expand x because autoencoder requires rgb image
                x = x.to(device)

                plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.show()
                exit(0)

                # Encode all training samples (This has to be done individually as calling the function on the batch
                # will cause weird encodings)
                x_encoded = torch.zeros(x.shape[0], image_channels, encoded_image_size, encoded_image_size).to(device)
                for i in range(x.shape[0]):
                    x_encoded[i] = autoencoder_model.encode_to_prequant(x[i:i+1])[0]

                # TODO: remove this
                #print(torch.min(x))
                #print(torch.max(x))
                #plt.imshow(x[0].permute(1, 2, 0).cpu().detach().numpy())
                #plt.title("Ground Truth")
                #plt.show()
                #print(torch.min(x_encoded))
                #print(torch.max(x_encoded))
                #plt.imshow(x_encoded[0].permute(1, 2, 0).cpu().detach().numpy())
                #plt.title("Encoded")
                #plt.show()

                loss = loss_fn(score_model, x_encoded, marginal_prob_std_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x_encoded.shape[0]
                num_items += x.shape[0]
                state['ema'].update(score_model.parameters())
                state['step'] += 1

            print('Average Loss: {:5f}'.format(avg_loss / num_items))

            torch.save(score_model.state_dict(), experiment_dir + "ckpt.pth")  # Update checkpoint after each epoch of training

            if epoch > n_skip_epochs:
                losses.append(avg_loss/num_items)
                plt.clf()
                plt.plot(np.arange(n_skip_epochs, len(losses) + n_skip_epochs), losses)
                plt.savefig(experiment_dir + "losses.jpg")
                plt.clf()

    # Sampling =========================================================================================================
    generate_samples(marginal_prob_std_fn, score_model, diffusion_coeff_fn, test_dataset,
                     autoencoder_model=autoencoder_model)


if __name__ == "__main__":
    main()
