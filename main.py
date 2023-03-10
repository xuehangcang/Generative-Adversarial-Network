import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# 超参数
epochs = 200
batch_size = 64
sample_size = 100
g_lr = 0.0001
d_lr = 0.0001
# 数据集
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    """生成器网络"""

    def __init__(self, sample_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(sample_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
        self.sample_size = sample_size

    def forward(self, batch_size: int):
        z = torch.randn(batch_size, self.sample_size).to(device)
        output = self.model(z)
        generated_images = output.reshape(batch_size, 1, 28, 28)
        return generated_images.to(device)


class Discriminator(nn.Module):
    """判别器网络"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )

    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        prediction = self.model(images.reshape(-1, 784))
        loss = f.binary_cross_entropy_with_logits(prediction, targets.to(device))
        return loss


def save_image_grid(epoch: int, images: torch.Tensor, ncol: int):
    """Save a grid of images to disk."""
    image_grid = make_grid(images, ncol)
    image_grid = image_grid.permute(1, 2, 0)
    image_grid = image_grid.cpu().numpy()
    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'generated_{epoch:03d}.jpg')
    plt.close()


# Real and fake labels
real_targets = torch.ones(batch_size, 1).to(device)
fake_targets = torch.zeros(batch_size, 1).to(device)

# Generator and Discriminator networks
generator = Generator(sample_size).to(device)
discriminator = Discriminator().to(device)

# Optimizers
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)

# Training loop
for epoch in range(epochs):

    d_losses = []
    g_losses = []

    for images, labels in tqdm(dataloader):
        # ===============================
        # Discriminator Network Training
        # ===============================

        # Loss with MNIST image inputs and real_targets as labels
        discriminator.train()
        d_loss = discriminator(images.to(device), real_targets)

        # Generate images in eval mode
        generator.eval()
        with torch.no_grad():
            generated_images = generator(batch_size)

        # Loss with generated image inputs and fake_targets as labels
        d_loss += discriminator(generated_images, fake_targets)

        # Optimizer updates the discriminator parameters
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============================
        # Generator Network Training
        # ===============================

        # Generate images in train mode
        generator.train()
        generated_images = generator(batch_size)

        # Loss with generated image inputs and real_targets as labels
        discriminator.eval()  # eval but we still need gradients
        g_loss = discriminator(generated_images, real_targets)

        # Optimizer updates the generator parameters
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Keep losses for logging
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    # Print average losses
    print(epoch, np.mean(d_losses), np.mean(g_losses))

    # Save images
    save_image_grid(epoch, generator(batch_size), ncol=8)

# Save the generator network
torch.save(generator.state_dict(), "generator.pth")
generator.eval()
with torch.no_grad():
    generated_images = generator(1)
    save_image_grid(200, generated_images, ncol=1)
