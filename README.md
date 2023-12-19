# Latent Diffusion Model
 Diffusion model used for reconstructing CT-scan images, trained in a lower dimensional latent space using an autoencoder.

# Dataset
 The dataset we used can be downloaded from the following github repo: https://github.com/swing-research/conditional-trumpets
 Make sure to rename the dataset path variable in main.py

# Autoencoder
 The autoencoder used for this experiment can be found in the following github repo: https://github.com/CompVis/latent-diffusion#model-zoo
 Make sure to place your downloaded model such that your directory structure looks as follows: ```models/vq-f4/model.ckpt```

# Training and Sampling
 In the ```scripts``` directory we have files with example usages of how to train and sample.
