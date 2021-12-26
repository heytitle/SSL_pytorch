from os import makedirs

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import resnet18

from models import BarlowTwins

# from optimizer import LARS
from trainer import SSL_Trainer
from utils import SSL_CIFAR10

from datetime import datetime


now = datetime.now()

slug = now.strftime("%m-%d-%Y--%H-%M-%S")

# Define hyperparameters
data_root = "./data"
save_root = f"./results/barlow_twins/{slug}"

print(f"Saving results at: {save_root}")

dl_kwargs = {"batch_size": 512, "shuffle": True, "num_workers": 2}

# Define data
ssl_data = SSL_CIFAR10(data_root, "SimCLR", dl_kwargs)

# general training params
train_params = {
    "save_root": save_root,
    "num_epochs": 200,
    "optimizer": SGD,
    "scheduler": CosineAnnealingLR,
    "warmup_epochs": 0,
    "iter_scheduler": True,
    "evaluate_at": [100, 200, 400, 600],
    "verbose": True,
}

# params of optimizer
## In Original Paper for Imagenet when using LARS Optimizer
# optim_params = {'lr':0.2 * dl_kwargs['batch_size']/256, 'weight_decay': 1.5e-6,
#                'exclude_bias_and_norm': True}

# from: lightly
optim_params = {"lr": 6e-2, "momentum": 0.9, "weight_decay": 5e-4}

# params of scheduler
scheduler_params = {
    "T_max": (train_params["num_epochs"] - train_params["warmup_epochs"])
}
# 'eta_min': 1e-3} in orginal implementation

# Set parameters for fitting linear protocoler
eval_params = {"lr": 1e-2, "num_epochs": 25, "milestones": [12, 20]}

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Print Device Type
if torch.cuda.is_available():
    print(f"Program running on {torch.cuda.get_device_name(device)}")
else:
    print("Program running on CPU")

# Create folder if it does not exists
makedirs(save_root, exist_ok=True)

# Define Model
resnet = resnet18()
barlow_twins = BarlowTwins(resnet, projector_hidden=(2048, 2048, 2048))

print(barlow_twins)

# Define Trainer
cifar10_trainer = SSL_Trainer(barlow_twins, ssl_data, device)

# Train
cifar10_trainer.train(
    **train_params,
    optim_params=optim_params,
    scheduler_params=scheduler_params,
    eval_params=eval_params,
)
