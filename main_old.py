import functools
import sys
import random
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.rcParams['figure.dpi'] = 600
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import warnings
from PIL import Image
from models import ScoreNet, ScoreUnet
from torch.optim import Adam
from ema import ExponentialMovingAverage
from tqdm import tqdm
from scipy import integrate
from torchvision.utils import make_grid
from skimage.transform import radon, iradon
from functions import *
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio

# PyTorch Arguments
if torch.cuda.device_count() > 1:
    torch.cuda.set_device(0)
torch_type = torch.float
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random Seeds
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Dataset Information
# dataset_path = "C:/Users/marko/Desktop/Bachelors Thesis/limited-CT_64/limited-CT/horizontal_snr25.0.npz"  # local PC
limited_CT_dataset_path = os.path.join("..", "datasets", "limited-CT_64", "limited-CT", "horizontal_snr25.0.npz")  # on sciCORE
inverse_scattering_dataset_path = os.path.join("..", "datasets", "scatterview", "inverse_scattering", "full",
                                               "scatter_data_full")
image_size = 64
batch_size = 32

# Experiment Folder variables
experiment_dir = "experiments/inverse_scattering_full_view/"  # dir for training session results
sampling_dir = "sampling_2/"  # dir used for storing sampling results (is within experiment_dir)
# if os.path.exists(experiment_dir):
#     raise Exception("Specified experiment directory already exists, use a different name")
# os.makedirs(experiment_dir)

# Training Arguments
num_workers_train = 4
num_workers_test = 4
n_epochs = 400
n_progress_samples = 1  # How often the sampler should be called during training
lr = 1e-4  # learning rate
losses = []
n_skip_epochs = 3  # Number of epochs at the beginning where the losses shouldn't be plotted

# Sampling Arguments
num_steps = 5000 # Number of sampling steps
sample_batch_size = 5  # needs to be at least 2 otherwise sampling won't work
n_posterior_samples = 100  # This needs to be at least 3
global_sigma = 25.0
global_tau = 0.5
n_angles = 90
theta_low = 0  # lower value for angles
theta_max = 180  # higher value for angles
angles = np.linspace(theta_low, theta_max, n_angles)
sigma_noise = 0.5
n_snr_comparison_images = 5
n_column_samples = 5  # needs to be at least 2 otherwise sampling won't work
n_columns = 7

# Program Arguments
skip_training = False
dataset_option = "Limited CT"  # can be "Limited CT" or "Inverse Scattering"
conditional_training = False
clean_training = False
include_gradient_descent = True
limited_view = False
image_channels = 1
additional_comments_sampling = ""
additional_comments_training = ""
visualisation_cmap = "gray"
visualize_noisy = False

# Checkers
if n_posterior_samples < 3:
    raise Exception("Number of posterior samples need to be at least 3")
if sample_batch_size < 2:
    raise Exception("Sample Batch size cannot be less than 2")
if n_column_samples < 2:
    raise Exception("Number of column samples cannot be less than 2")


class InverseScatteringLoader(Dataset):
    def __init__(self, path, option, size=(128, 128)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.data_gt = np.load(os.path.join(path, "gt.npy"))
        self.data_y = np.load(os.path.join(path, "y.npy"))
        self.data_gt_test = np.load(os.path.join(path, "gt_test.npy"))
        self.data_y_test = np.load(os.path.join(path, "y_test.npy"))

        max_gt = np.max([self.data_gt.max(), self.data_gt_test.max()])
        min_gt = np.min([self.data_gt.min(), self.data_gt_test.min()])
        max_y = np.max([self.data_y.max(), self.data_y.max()])
        min_y = np.min([self.data_y_test.min(), self.data_y_test.min()])

        self.data_gt = (self.data_gt - min_gt) / (max_gt - min_gt)
        self.data_y = (self.data_y - min_y) / (max_y - min_y)
        self.data_gt_test = (self.data_gt_test - min_gt) / (max_gt - min_gt)
        self.data_y_test = (self.data_y_test - min_y) / (max_y - min_y)

        if option == "train":
            # Reshape the two arrays
            self.data_gt = self.data_gt[:, np.newaxis, :, :]
            self.data_y = self.data_y[:, np.newaxis, :, :]
            # Combine them into one array. The resulting array has shape (N, 2, S, S), where N is the number of images
            # S is the image size and the second axis is for ground truth (0) or y (1)
            self.data = np.concatenate((self.data_gt, self.data_y), axis=1)

        elif option == "test":
            # Reshape the two arrays
            self.data_gt = self.data_gt_test[:, np.newaxis, :, :]
            self.data_y = self.data_y_test[:, np.newaxis, :, :]

            # Combine them into one array. The resulting array has shape (2, N, 1, S, S), where N is the number of images
            # S is the image size and the second axis is for ground truth (0) or y (1)
            self.data = np.concatenate((self.data_gt, self.data_y), axis=1)
        else:
            raise Exception("Incorrect option for Inverse scatter Loader chosen. Try train or test")

        self.im_size = size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        return torch.from_numpy(img)


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
        self.meshgrid = get_mgrid(size[0])

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, index):
        img = self.img_dataset[index][:, :, 0]  # Weird array access bcs dataset image has shape (64, 64, 1)
        img = Image.fromarray(np.uint8(img), 'L')  # Transforms to PIL image for self.transform
        img = self.transform(img)
        return img


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
    if conditional_training:
        z[:, 1, :, :] = 0  # we don't want to add noise to the conditional image
    std = marginal_prob_std(random_t, device=x.device)

    perturbed_x = x + z * std[:, None, None, None]

    score = model(perturbed_x, random_t)

    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z[:, :1])**2, dim=(1, 2, 3)))
    return loss


def euler_maruyama_sampler(score_model,
                           y,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
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
    init_x = z
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    # fbp reconstruction of x in the case of conditionally trained NN
    x_fbp = torch.zeros_like(x)

    if dataset_option == "limitedCT":
        if conditional_training:
            for i in range(sample_batch_size):
                x_fbp[i, 0] = torch.from_numpy(iradon(y[i], angles, circle=False))
    elif dataset_option == "scatterview":
        for i in range(sample_batch_size):
            x_fbp[i, 0] = torch.from_numpy(y[i])

    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step, device=device)

            if conditional_training:
                mean_x = x + (g ** 2)[:, None, None, None] * score_model(torch.cat((x, x_fbp), dim=1), batch_time_step)\
                         * step_size
            else:
                mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size

            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

            # Add filtered back projection to make sampler conditional
            if include_gradient_descent:
                for im_index in range(batch_size):
                    x_numpy = x[im_index, 0, :, :].cpu().detach().numpy()  # converting pytorch tensor to numpy array
                    forward_x = radon(x_numpy, angles, circle=False)  # this is A(Phi)z^{~}_{t+1}
                    new_x = x_numpy - tau * (iradon(forward_x - y[im_index], angles, circle=False))
                    x[im_index, 0, :, :] = torch.from_numpy(new_x)  # convert back to pytorch tensor
                    # Do not include any noise in the last sampling step.
    return mean_x


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
        axs[i, 2].imshow(samples[0, i, 0].cpu().detach().numpy(), cmap=visualisation_cmap)
        axs[i, 3].imshow(samples[1, i, 0].cpu().detach().numpy(), cmap=visualisation_cmap)
        axs[i, 4].imshow(samples[2, i, 0].cpu().detach().numpy(), cmap=visualisation_cmap)
        axs[i, 5].imshow(samples_std[0, i, 0], cmap=visualisation_cmap)
        axs[i, 6].imshow(samples_mean[0, i, 0], cmap=visualisation_cmap)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(experiment_dir + sampling_dir + figure_title + '.jpg', dpi=600)


def snr_comparison_visualisation(image1, image2, title1, title2, figure_title):
    plt.clf()
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image1, cmap=visualisation_cmap)
    ax1.set_title(title1)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image2, cmap=visualisation_cmap)
    ax2.set_title(title2)
    ax2.axis("off")

    snr = compute_snr(image1, image2)
    fig.text(0.5, 0.1, "SNR = " + str(snr), ha="center")
    plt.savefig(experiment_dir + sampling_dir + figure_title + '.jpg')


def generate_samples(marginal_prob_std_fn, score_model, diffusion_coeff_fn, im_dataset):
    global device
    t = torch.ones(sample_batch_size, device=device)

    # z_init, has multiple channels so that multiple different noises for posterior samples are generated
    z_init = torch.randn(sample_batch_size, n_posterior_samples, image_size, image_size, device=device) \
        * marginal_prob_std_fn(t, device=device)[:, None, None, None]

    # Load the pre-trained checkpoint from disk.
    device = 'cuda'  # @param ['cuda', 'cpu'] {'type':'string'}
    ckpt = torch.load(experiment_dir + 'ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    # Get true images
    true_images = []
    for i in range(sample_batch_size):
        true_images.append(im_dataset.__getitem__(i).cpu().detach().numpy()[0])

    # Get measurements from training set to make sampling conditional
    y = []
    y_noisy = []
    for i in range(sample_batch_size):
        if dataset_option == "limitedCT":
            y.append(radon(true_images[i], angles, circle=False))  # im from dataset that makes it conditional
            y_noisy.append((torch.from_numpy(y[i]) + sigma_noise *
                            torch.randn_like(torch.from_numpy(y[i]))).cpu().detach().numpy())
        elif dataset_option == "scatterview":
            y.append(im_dataset.__getitem__(i).cpu().detach().numpy()[1])

    # Compute FBP reconstruction
    samples_fbp_noisy = []
    samples_fbp_clean = []
    if dataset_option == "limitedCT":
        for i in range(sample_batch_size):
            samples_fbp_clean.append(iradon(y[i], angles, circle=False))
            samples_fbp_noisy.append(iradon(y_noisy[i], angles, circle=False))

    # Get samples from sampler =========================================================================================
    samples_noisy = []
    samples_clean = []
    samples = []
    for i in range(n_posterior_samples):
        if dataset_option == "limitedCT":
            posterior_samples_noisy = euler_maruyama_sampler(score_model, y_noisy, marginal_prob_std_fn,
                                                             diffusion_coeff_fn, sample_batch_size, device=device,
                                                             z=z_init[:, i:(i+1)], tau=global_tau)
            posterior_samples_noisy = posterior_samples_noisy.clamp(0.0, 1.0)
            samples_noisy.append(posterior_samples_noisy)

            posterior_samples_clean = euler_maruyama_sampler(score_model, y, marginal_prob_std_fn,
                                                             diffusion_coeff_fn, sample_batch_size, device=device,
                                                             z=z_init[:, i:(i+1)], tau=global_tau)
            posterior_samples_clean = posterior_samples_clean.clamp(0.0, 1.0)
            samples_clean.append(posterior_samples_clean)
        elif dataset_option == "scatterview":
            posterior_samples = euler_maruyama_sampler(score_model, y, marginal_prob_std_fn,
                                                             diffusion_coeff_fn, sample_batch_size, device=device,
                                                             z=z_init[:, i:(i+1)], tau=global_tau)
            samples.append(posterior_samples)

    # Get STD of samples ===============================================================================================
    # Noisy Samples
    if dataset_option == "limitedCT":
        samples_noisy = torch.stack(samples_noisy)
        # samples tensor has dimensions n_posterior_samples x sample_batch_size x n_channels x im_size x im_size
        samples_noisy_std = torch.std(samples_noisy, dim=0, keepdim=True)
        samples_noisy_std = samples_noisy_std.cpu().detach().numpy()
        rescaled_samples_noisy_std = (samples_noisy_std - np.min(samples_noisy_std)) / \
                                     (np.max(samples_noisy_std) - np.min(samples_noisy_std))

        # Clean samples
        samples_clean = torch.stack(samples_clean)
        # samples tensor has dimensions n_posterior_samples x sample_batch_size x n_channels x im_size x im_size
        samples_clean_std = torch.std(samples_clean, dim=0, keepdim=True)
        samples_clean_std = samples_clean_std.cpu().detach().numpy()
        rescaled_samples_clean_std = (samples_clean_std - np.min(samples_clean_std)) / \
                                     (np.max(samples_clean_std) - np.min(samples_clean_std))

    elif dataset_option == "scatterview":
        samples = torch.stack(samples)
        samples_std = torch.std(samples, dim=0, keepdim=True)
        samples_std = samples_std.cpu().detach().numpy()
        rescaled_samples_std = (samples_std - np.min(samples_std)) / \
                                     (np.max(samples_std) - np.min(samples_std))

    # Get Mean of all posterior samples ================================================================================
    if dataset_option == "limitedCT":
        samples_noisy_mean = torch.mean(samples_noisy, dim=0, keepdim=True).cpu().detach().numpy()
        samples_clean_mean = torch.mean(samples_clean, dim=0, keepdim=True).cpu().detach().numpy()

    elif dataset_option == "scatterview":
        samples_mean = torch.mean(samples, dim=0, keepdim=True).cpu().detach().numpy()

    # Compute different SNR's for the first few images and visualize them ==============================================
    if dataset_option == "limitedCT":
        for i in range(n_snr_comparison_images):
            # Between FBP and True image
            snr_comparison_visualisation(samples_fbp_clean[i], true_images[i], "FBP reconstruction on noiseless y",
                                         "True image", "FBP_vs_True_" + str(i))
            # Between Our estimation vs True image
            snr_comparison_visualisation(samples_noisy_mean[0, i, 0], true_images[i],
                                         "Reconstruction from Noisy y",
                                         "True image", "Noisy_reconstruction_vs_True_" + str(i))
            snr_comparison_visualisation(samples_fbp_clean[i], samples_fbp_noisy[i], "FBP recon on clean y",
                                         "FBP recon on noisy y", "fbpclean_vs_fbpnoisy_" + str(i))
            snr_comparison_visualisation(samples_clean_mean[0, i, 0], true_images[i],
                                         "Reconstruction from clean y",
                                         "True image", "Clean_reconstruction_vs_True_" + str(i))

    elif dataset_option == "scatterview":
        for i in range(n_snr_comparison_images):
            snr_comparison_visualisation(samples_mean[0, i, 0], true_images[i],
                                         "Reconstruction from y",
                                         "True image", "Reconstruction_vs_True_" + str(i))

    # Column Visualization =============================================================================================
    if dataset_option == "limitedCT":
        column_visualization(true_images, samples_fbp_noisy, samples_noisy, rescaled_samples_noisy_std, samples_noisy_mean,
                                 "FBP, noisy y", "Noisy samples", "column_visualization_noisy")
        column_visualization(true_images, samples_fbp_clean, samples_clean, rescaled_samples_clean_std, samples_clean_mean,
                                 "FBP, clean y", "Clean samples", "column_visualization_clean")

    elif dataset_option == "scatterview":
        column_visualization(true_images, y, samples, rescaled_samples_std, samples_mean, "y", "Samples",
                             "column_visaulization")

    # Get average SNR and PSNR from samples ============================================================================
    noisy_samples_average_snr = 0
    noisy_samples_average_psnr = 0
    clean_samples_average_snr = 0
    clean_samples_average_psnr = 0
    samples_average_snr = 0
    samples_average_psnr = 0

    if dataset_option == "limitedCT":
        for i in range(sample_batch_size):
            noisy_samples_average_snr += compute_snr(samples_noisy_mean[0, i, 0], true_images[i])
            noisy_samples_average_psnr += peak_signal_noise_ratio(samples_noisy_mean[0, i, 0],
                                                                  true_images[i])

            clean_samples_average_snr += compute_snr(samples_clean_mean[0, i, 0], true_images[i])
            clean_samples_average_psnr += peak_signal_noise_ratio(samples_clean_mean[0, i, 0],
                                                                  true_images[i])
        noisy_samples_average_snr = noisy_samples_average_snr/sample_batch_size
        noisy_samples_average_psnr = noisy_samples_average_psnr / sample_batch_size

        clean_samples_average_snr = clean_samples_average_snr/sample_batch_size
        clean_samples_average_psnr = clean_samples_average_psnr / sample_batch_size

    elif dataset_option == "scatterview":
        for i in range(sample_batch_size):
            samples_average_snr += compute_snr(samples_mean[0, i, 0], true_images[i])
            samples_average_psnr += peak_signal_noise_ratio(samples_mean[0, i, 0],
                                                            true_images[i])
        samples_average_snr = samples_average_snr/sample_batch_size
        samples_average_psnr = samples_average_psnr / sample_batch_size

    # Get average SNR and PSNR between noisy and clean FBP reconstruction ==============================================
    if dataset_option == "limitedCT":
        y_clean_vs_noisy_average_snr = 0
        for i in range(sample_batch_size):
            y_clean_vs_noisy_average_snr += compute_snr(y[i], y_noisy[i])
        y_clean_vs_noisy_average_snr = y_clean_vs_noisy_average_snr/sample_batch_size

    elif dataset_option == "scatterview":
        pass

    # Write experiment results to txt file =============================================================================
    file = open(experiment_dir + sampling_dir + "experiment_results" + ".txt", "w")
    current_datetime = datetime.now()

    if dataset_option == "limitedCT":
        file.write("Results from sampling done on " + str(current_datetime) + ": \n" +
                   "Noise parameter value: " + str(sigma_noise) + "\n" +
                   "Batch size: " + str(sample_batch_size) + "\n" +
                   "Number of sampling steps: " + str(num_steps) + "\n" +
                   "Number of posterior samples: " + str(n_posterior_samples) + "\n" +
                   "Average SNR between noisy sinogram (noisy y) and clean sinogram (clean y) for batch: "
                   + str(y_clean_vs_noisy_average_snr) + "\n" +
                   "Average SNR between mean of noisy posterior samples and true images for batch: "
                   + str(noisy_samples_average_snr) + "\n" +
                   "Average PSNR between mean of noisy posterior samples and true images for batch: "
                   + str(noisy_samples_average_psnr) + "\n" +
                   "Average SNR between mean of clean posterior samples and true images for batch: "
                   + str(clean_samples_average_snr) + "\n" +
                   "Average PSNR between mean of clean posterior samples and true images for batch: "
                   + str(clean_samples_average_psnr) + "\n" +
                   "FBP reconstruction number of angles: " + str(n_angles) + "\n" +
                   "FBP reconstruction angle range: " + str(theta_low) + " to " + str(theta_max) + "\n" +
                   "Gradient descent step in sampling included: " + str(include_gradient_descent) + "\n" +
                   additional_comments_sampling)

    elif dataset_option == "scatterview":
        file.write("Results from sampling done on " + str(current_datetime) + ": \n" +
                   "Batch size: " + str(sample_batch_size) + "\n" +
                   "Number of sampling steps: " + str(num_steps) + "\n" +
                   "Number of posterior samples: " + str(n_posterior_samples) + "\n" +
                   "Average SNR between mean of posterior samples and true images for batch: "
                   + str(samples_average_snr) + "\n" +
                   "Average PSNR between mean of posterior samples and true images for batch: "
                   + str(samples_average_psnr) + "\n" +
                   additional_comments_sampling)


def main():
    # Load dataset ==========================================================================
    if dataset_option == "limitedCT":
        train_dataset = LimitedCT64X64Loader(path=limited_CT_dataset_path, option="train",
                                             size=(image_size, image_size))
        test_dataset = LimitedCT64X64Loader(path=limited_CT_dataset_path, option="test",
                                            size=(image_size, image_size))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers_train,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers_train,
                                                  shuffle=True)
    elif dataset_option == "scatterview":
        train_dataset = InverseScatteringLoader(path=inverse_scattering_dataset_path, option="train",
                                                size=(image_size, image_size))
        test_dataset = InverseScatteringLoader(path=inverse_scattering_dataset_path, option="test",
                                               size=(image_size, image_size))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers_train,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers_train,
                                                  shuffle=True)
    else:
        raise Exception("Wrong dataset option variable value. Has to be either 'scatterview' or 'limitedCT'. "
                        "Currently is: " + str(dataset_option))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=global_sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=global_sigma)

    # Define the model ======================================================================
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, image_channels=image_channels) # TODO does this need to be changed for scatterview?
    score_model = score_model.to(device)

    num_param = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    print('---> Number of trainable parameters of Autoencoder: {}'.format(num_param))

    # Training ==============================================================================
    if not skip_training:
        optimizer = Adam(score_model.parameters(), lr=lr)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        pbar = tqdm(range(n_epochs))
        sample_counter = 1
        epoch_counter = 0
        counter = 0
        for epoch in pbar:
            avg_loss = 0.
            num_items = 0

            train_counter = 0
            for x in train_loader:

                if counter < 4:  # Saves the first three samples into files for debugging
                    name = experiment_dir + "train_sample" + str(counter) + ".jpg"
                    plt.imshow(x[0, 0])
                    plt.savefig(name)
                    counter += 1

                x = x.to(device).requires_grad_(True)

                if dataset_option == "limitedCT":
                    if conditional_training:
                        x_fbp = torch.clone(x)
                        for i in range(x.shape[0]):
                            if clean_training:
                                x_forward_clean = radon(x[i, 0, :, :].detach().cpu().numpy(), angles, circle=False)
                                x_fbp[i, 0, :, :] = torch.from_numpy(iradon(x_forward_clean, angles, circle=False))
                            else:
                                x_forward_clean = radon(x[i, 0, :, :].detach().cpu().numpy(), angles, circle=False)
                                noise = sigma_noise * np.random.randn(x_forward_clean.shape[0], x_forward_clean.shape[1])
                                x_forward = x_forward_clean + noise
                                x_fbp[i, 0, :, :] = torch.from_numpy(iradon(x_forward, angles, circle=False))
                        x = torch.cat((x, x_fbp), dim=1)   # concatenate the fbp reconstruction to x
                elif dataset_option == "scatterview":
                    pass

                loss = loss_fn(score_model, x, marginal_prob_std_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                state['ema'].update(score_model.parameters())
                state['step'] += 1

            # Print the averaged training loss so far.
            print('Average Loss: {:5f}'.format(avg_loss / num_items))

            # Add average loss to losses for plotting after training
            if epoch_counter > n_skip_epochs:
                losses.append(avg_loss / num_items)
            epoch_counter += 1
            plt.clf()
            x_values = np.arange(n_skip_epochs, len(losses) + n_skip_epochs)
            plt.plot(x_values, losses)
            plt.savefig(experiment_dir + "losses.jpg")
            plt.clf()

            # Update the checkpoint after each epoch of training.
            torch.save(score_model.state_dict(), experiment_dir + 'ckpt.pth')

            # Generate progress samples during training
            # if epoch > n_epochs/n_progress_samples * sample_counter:
            #    sample_name = "sample" + str(sample_counter)
            #    generate_samples(marginal_prob_std_fn, score_model, diffusion_coeff_fn)
            #    sample_counter += 1

        # Information about the trained checkpoint
        file = open(experiment_dir + "training_parameters" + ".txt", "w")
        now = datetime.now()

        if dataset_option == "limitedCT":
            file.write("Parameters for training done on " + str(now) + ": \n" +
                        "Noise parameter value: " + str(sigma_noise) + "\n" +
                        "Conditionally trained: " + str(conditional_training) + "\n" +
                        "Trained with no noise: " + str(clean_training) + "\n" +
                        "FBP reconstruction number of angles: " + str(n_angles) + "\n" +
                        "FBP reconstruction angles :" + str(theta_low) + " to " + str(theta_max) + "\n" +
                        additional_comments_training)

        elif dataset_option == "scatterview":
            file.write("Parameters for training done on " + str(now) + ": \n" +
                        "Conditionally trained: " + str(conditional_training) + "\n" +
                        additional_comments_training)

    # Sampling ==============================================================================
    generate_samples(marginal_prob_std_fn, score_model, diffusion_coeff_fn, test_dataset)


def string_match(array, pattern):
    for string in array:
        match = re.search(pattern, string)
        if match:
            return match.group(1)
    return None


if __name__ == "__main__":
    pattern_experiment_dir = r'^exdir:(.*)'
    pattern_sampling_dir = r'^sampledir:(.*)'

    if sys.argv[1] == "limitedCT":
        dataset_option = "limitedCT"
        image_size = 64

        if "visualizenoisy" in sys.argv:
            visualize_noisy = True
        if "seismic" in sys.argv:
            visualisation_cmap = "seismic"
        if "sample" in sys.argv:
            skip_training = True
        if "conditional" in sys.argv:
            image_channels = 2
            conditional_training = True
        if "clean" in sys.argv:
            clean_training = True
        if "skipgrad" in sys.argv:
            include_gradient_descent = False
        if "limitedview" in sys.argv:
            limited_view = True
            theta_max = 90
            angles = np.linspace(theta_low, theta_max, n_angles)
        else:
            theta_max = 180
            angles = np.linspace(theta_low, theta_max, n_angles)

        experiment_dir_found = string_match(sys.argv, pattern_experiment_dir)
        if experiment_dir_found:
            experiment_dir = "experiments/" + experiment_dir_found + "/"

        sampling_dir_found = string_match(sys.argv, pattern_sampling_dir)
        if sampling_dir_found:
            sampling_dir = sampling_dir_found + "/"

        print("\nProgram arguments for this session are: \n\n" +
              "dataset_option" + str(dataset_option) + "\n" +
              "skip_training: " + str(skip_training) + "\n" +
              "conditional_training: " + str(conditional_training) + "\n" +
              "clean_training: " + str(clean_training) + "\n" +
              "include_gradient_descent: " + str(include_gradient_descent) + "\n" +
              "theta_max (Limited view sinograms): " + str(theta_max) + "\n" +
              "experiment_dir: " + str(experiment_dir) + "\n" +
              "sampling_dir: " + str(sampling_dir) + "\n" +
              "visualisation_cmap: " + str(visualisation_cmap) + "\n" +
              "visualize_noisy: " + str(visualize_noisy))

    elif sys.argv[1] == "scatterview":
        dataset_option = "scatterview"
        image_size = 128

        include_gradient_descent = False
        conditional_training = True
        image_channels = 2

        if "seismic" in sys.argv:
            visualisation_cmap = "seismic"
        if "sample" in sys.argv:
            skip_training = True
        if "conditional" in sys.argv:
            warnings.warn("'conditional' keyword has no effect when using the scatterview dataset.")
        if "clean" in sys.argv:
            warnings.warn("'clean' keyword has no effect when using the scatterview dataset.")
        if "skipgrad" in sys.argv:
            warnings.warn("'skipgrad' keyword has no effect when using the scattreview dataset.")
        if "limitedview" in sys.argv:
            inverse_scattering_dataset_path = os.path.join("..", "datasets", "scatterview", "inverse_scattering",
                                                           "limited-view", "scatter_data_BP_top")
            limited_view = True

        experiment_dir_found = string_match(sys.argv, pattern_experiment_dir)
        if experiment_dir_found:
            experiment_dir = "experiments/" + experiment_dir_found + "/"

        sampling_dir_found = string_match(sys.argv, pattern_sampling_dir)
        if sampling_dir_found:
            sampling_dir = sampling_dir_found + "/"

        print("\nProgram arguments for this session are: \n\n" +
              "dataset_option: " + str(dataset_option) + "\n" +
              "skip_training: " + str(skip_training) + "\n" +
              "conditional_training: " + str(conditional_training) + "\n" +
              "include_gradient_descent: " + str(include_gradient_descent) + "\n" +
              "experiment_dir: " + str(experiment_dir) + "\n" +
              "sampling_dir: " + str(sampling_dir) +
              "visualisation_cmap: " + str(visualisation_cmap) + "\n")

    else:
        raise Exception("Incorrect first argument, has to be either 'limitedCT' or 'scatterview'.")

    os.makedirs(experiment_dir + sampling_dir)

    main()
