# Library imports
import torch
import os
import sys
import functools
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage.transform import radon, iradon
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from ema import ExponentialMovingAverage
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio

# Local file imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.psld import DDIMSampler
from functions import marginal_prob_std
from functions import diffusion_coeff
from models.models import ScoreNet
from functions import compute_snr


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

# Pytorch
device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")
print("Torch Device: " + str(device))

# Training
batch_size = 32
num_workers_train = 4
num_workers_test = 4
global_sigma = 25.0
n_epochs = 100
learning_rate = 1e-3
losses = []
n_skip_epochs = 4  # Number of epochs to skip for plotting the training
image_channels = 6
experiment_dir = "experiments/experiment6/"  # Directory where checkpoint is

# Sampling
num_steps = 100  # Number of sampling steps
sample_batch_size = 20
n_angles = 90
theta_low = 0  # lower value for angles
theta_max = 180  # higher value for angles
angles = np.linspace(theta_low, theta_max, n_angles)
n_posterior_samples = 10
global_tau = 0.5
sampling_dir = "sampling_final/"  # Directory for current sampling instance
n_column_samples = 5
n_columns = 7
visualisation_cmap = "gray"

# Program arguments:
conditional_training = True
include_gradient_descent = False
skip_training = True
additional_comments_training = ""
additional_comments_sampling = ""
limited_view = False

# Checkers
if n_posterior_samples < 3:
    raise Exception("Number of posterior samples need to be at least 3")
if sample_batch_size < 2:
    raise Exception("Sample Batch size cannot be less than 2")
if n_column_samples < 2:
    raise Exception("Number of column samples cannot be less than 2")


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


def column_visualization(true_images, samples_fbp, samples, samples_std, samples_mean, fbp_title, sample_title,
                         figure_title):
    plt.close("all")
    fig, axs = plt.subplots(nrows=n_column_samples, ncols=n_columns, figsize=(n_columns, n_column_samples + 0.5),
                            dpi=600)
    matplotlib.rcParams.update({'font.size': 7})
    matplotlib.rcParams.update({'figure.dpi': 600})

    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].set_title(fbp_title)
    axs[0, 2].set_title(sample_title)
    axs[0, 5].set_title("STD")
    axs[0, 6].set_title("MEAN")

    for i in range(n_column_samples):
        axs[i, 0].imshow(true_images[i], cmap=visualisation_cmap)
        axs[i, 1].imshow(samples_fbp[i], cmap=visualisation_cmap)
        axs[i, 2].imshow(samples[0, i].permute(1, 2, 0).cpu().detach().numpy(), cmap=visualisation_cmap)
        axs[i, 3].imshow(samples[1, i].permute(1, 2, 0).cpu().detach().numpy(), cmap=visualisation_cmap)
        axs[i, 4].imshow(samples[2, i].permute(1, 2, 0).cpu().detach().numpy(), cmap=visualisation_cmap)
        axs[i, 5].imshow(samples_std[0, i, 0], cmap=visualisation_cmap)
        axs[i, 6].imshow(samples_mean[0, i, 0], cmap=visualisation_cmap)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(experiment_dir + sampling_dir + figure_title + '.jpg', dpi=600)


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


def loss_fn(model, x, marginal_prob_std, autoencoder_model, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      autoencoder_model: autoencoder used to decode and encode into latent space
      eps: A tolerance value for numerical stability.
    """

    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    if conditional_training:
        z[:, 3:6, :, :] = 0  # we don't want to add noise to the conditioning input
    std = marginal_prob_std(random_t, device=x.device)

    perturbed_x = x + z * std[:, None, None, None]

    score = model(perturbed_x, random_t)

    # z is [:, 0:3] because we don't want to add noise to the conditioning input
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z[:, :3])**2, dim=(1, 2, 3)))
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
    x.requires_grad = True

    x_fbp = torch.zeros_like(x)
    for im_index in range(x.shape[0]):
        for i in range(3):
            x_fbp[im_index:im_index+1] = autoencoder_model.encode_to_prequant(
                torch.from_numpy(iradon(y[im_index], angles, circle=False))[None, None, :, :].repeat(1, 3, 1, 1).to(device))

    #with (torch.no_grad()):
    for time_step in tqdm(time_steps):
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step, device=device)

        if conditional_training:
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(torch.cat((x, x_fbp), dim=1),
                                                                     batch_time_step) * step_size
        else:
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size

        x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

        # Add filtered back projection to make sampler conditional
        if include_gradient_descent:
            for im_index in range(batch_size):
                #x_external = torch.zeros(x.shape[1], x.shape[2], x.shape[3])
                #for i in range(3):  # for all image channels
                #    x_external[i] = torch.from_numpy(iradon(radon(x[im_index, i].detach().cpu().numpy(),
                #                                                  angles, circle=False) -
                #                                            y[im_index][i].detach().cpu().numpy(),
                #                                            angles, circle=False))
                #x_external = x_external.to(device)
                #x[im_index].backward(x_external)
                for i in range(3):  # all image channels
                    x[im_index, i] = (x[im_index, i] - tau * torch.from_numpy(
                        iradon(radon(x[im_index, i].detach().cpu().numpy(), angles, circle=False)
                               - y[im_index][i].detach().cpu().numpy(), angles, circle=False)).to(device))
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
        true_images.append(im_dataset.__getitem__(i)[0][None].repeat(3, 1, 1).permute(1, 2, 0).cpu().detach().numpy())
        #true_images.append(im_dataset.__getitem__(i).cpu().detach().numpy()[0])

    # Get measurements from training set to make sampling conditional
    y = []
    y_encoded = []  # we want to have the y measurement in the latent space domain as well here
    for i in range(sample_batch_size):
        y.append(radon(true_images[i][:, :, 0], angles, circle=False))  # im from dataset that makes it conditional
        image = true_images[i][:, :, 0][None, None, :, :]
        image = torch.from_numpy(image).repeat(1, 3, 1, 1).to(device)
        encoded = autoencoder_model.encode_to_prequant(image)
        # you can get 23 by looking at the shape after radon
        y_encoding = torch.zeros(3, 23, n_angles)
        for j in range(3):  # all image channels
            y_encoding[j] = torch.from_numpy(radon(encoded.detach().cpu().numpy()[0, 0], angles, circle=False))
        y_encoded.append(y_encoding)

    # Compute FBP reconstruction
    samples_fbp_clean = []
    for i in range(sample_batch_size):
        samples_fbp_clean.append(iradon(y[i], angles, circle=False))

    encoded_samples_clean = []
    for i in range(n_posterior_samples):
        posterior_samples_clean = euler_maruyama_sampler(score_model,
                                                         y,
                                                         marginal_prob_std_fn,
                                                         diffusion_coeff_fn,
                                                         batch_size=sample_batch_size,
                                                         autoencoder_model=autoencoder_model,
                                                         device=device,
                                                         z=z_init[i], tau=global_tau)
        encoded_samples_clean.append(posterior_samples_clean)

    samples_clean = torch.zeros(n_posterior_samples, sample_batch_size, 3, image_size, image_size)
    for i in range(n_posterior_samples):
        for j in range(sample_batch_size):
            samples_clean[i, j:j+1] = autoencoder_model.decode(encoded_samples_clean[i][j:j+1])
    samples_clean = torch.clamp(samples_clean, min=0, max=1)

    # Get STD of samples ===============================================================================================
    # samples tensor has dimensions n_posterior_samples x sample_batch_size x n_channels x im_size x im_size
    samples_clean_std = torch.std(samples_clean, dim=0, keepdim=True)
    samples_clean_std = samples_clean_std.cpu().detach().numpy()
    rescaled_samples_clean_std = (samples_clean_std - np.min(samples_clean_std)) / \
                                 (np.max(samples_clean_std) - np.min(samples_clean_std))

    # Get Mean of all posterior samples ================================================================================
    samples_clean_mean = torch.mean(samples_clean, dim=0, keepdim=True).cpu().detach().numpy()

    column_visualization(true_images, samples_fbp_clean, samples_clean, rescaled_samples_clean_std, samples_clean_mean,
                         "FBP, y", "Samples", "column_visualization")

    # Compute average SNR and PSNR for samples =========================================================================
    samples_average_snr = 0
    samples_average_psnr = 0
    for i in range(sample_batch_size):
        samples_average_snr += compute_snr(torch.from_numpy(true_images[i]).permute(2, 0, 1).detach().cpu().numpy(),
                                           samples_clean_mean[0, i])
        samples_average_psnr += peak_signal_noise_ratio(
            torch.from_numpy(true_images[i]).permute(2, 0, 1).detach().cpu().numpy(),
            samples_clean_mean[0, i])
    samples_average_psnr = samples_average_psnr / sample_batch_size
    samples_average_snr = samples_average_snr / sample_batch_size

    print("Sampling Results: " + "\n"
          "Average PSNR for batch: " + str(samples_average_psnr) + "\n" +
          "Average SNR for batch: " + str(samples_average_snr) + "\n")


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

                # Encode all training samples (This has to be done individually as calling the function on the batch
                # will cause weird encodings)
                x_encoded = torch.zeros(x.shape[0], image_channels, encoded_image_size, encoded_image_size).to(device)
                for i in range(x.shape[0]):
                    x_encoded[i][0:3] = autoencoder_model.encode_to_prequant(x[i:i+1])[0]

                if conditional_training:  # We add the fbp reconstruction of x to the channels 3-5 (conditioning input)
                    for i in range(x.shape[0]):
                        x_fbp = iradon(radon(x[i, 0].detach().cpu().numpy(), angles, circle=False), angles, circle=False)
                        x_fbp = torch.from_numpy(x_fbp)
                        x_fbp = x_fbp[None, None, :, :].repeat(1, 3, 1, 1).to(device)
                        x_fbp_encoded = autoencoder_model.encode_to_prequant(x_fbp)
                        x_encoded[i][3:6] = x_fbp_encoded[0]  # add the conditioning input

                loss = loss_fn(score_model, x_encoded, marginal_prob_std_fn, autoencoder_model)
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

        # Information about the trained checkpoint
        file = open(experiment_dir + "training_parameters" + ".txt", "w")
        now = datetime.now()

        file.write("Parameters for training done on " + str(now) + ": \n" +
                   "Conditionally trained: " + str(conditional_training) + "\n" +
                   "FBP reconstruction number of angles: " + str(n_angles) + "\n" +
                   "FBP reconstruction angles :" + str(theta_low) + " to " + str(theta_max) + "\n" +
                   "Number of epochs: " + str(n_epochs) + "\n" +
                   additional_comments_training)

    # Sampling =========================================================================================================
    generate_samples(marginal_prob_std_fn, score_model, diffusion_coeff_fn, test_dataset,
                     autoencoder_model=autoencoder_model)


def string_match(array, pattern):
    for string in array:
        match = re.search(pattern, string)
        if match:
            return match.group(1)
    return None


if __name__ == "__main__":
    pattern_experiment_dir = r'^exdir:(.*)'
    pattern_sampling_dir = r'^sampledir:(.*)'

    if "sample" in sys.argv:
        skip_training = True
    if "conditional" in sys.argv:
        image_channels = 6
        conditional_training = True
    if "limitedview" in sys.argv:  # Limited view just changes the viewing angles from 0-180 to 0-90
        limited_view = True
        theta_max = 90
        angles = np.linspace(theta_low, theta_max, n_angles)

    experiment_dir_found = string_match(sys.argv, pattern_experiment_dir)
    if experiment_dir_found:
        experiment_dir = "experiments/" + experiment_dir_found + "/"

    sampling_dir_found = string_match(sys.argv, pattern_sampling_dir)
    if sampling_dir_found:
        sampling_dir = sampling_dir_found + "/"

    print("\nProgram arguments for this session are: \n\n" +
          "skip_training: " + str(skip_training) + "\n" +
          "conditional_training: " + str(conditional_training) + "\n" +
          "include_gradient_descent: " + str(include_gradient_descent) + "\n" +
          "theta_max (Limited view sinograms): " + str(theta_max) + "\n" +
          "experiment_dir: " + str(experiment_dir) + "\n" +
          "sampling_dir: " + str(sampling_dir) + "\n" +
          "visualisation_cmap: " + str(visualisation_cmap) + "\n")

    main()
