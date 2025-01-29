import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.image_files = os.listdir(low_res_dir)  # Assuming both folders have the same filenames

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        low_res_name = os.path.join(self.low_res_dir, self.image_files[idx])  # Path for low_res image
        high_res_name = os.path.join(self.high_res_dir, self.image_files[idx])  # Path for high_res image

        low_res_image = Image.open(low_res_name).convert("L")  # Convert to grayscale
        high_res_image = Image.open(high_res_name).convert("L")  # Convert to grayscale

        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image

# Residual Dense Block (RRDB)
class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out

# Generator
class Generator(nn.Module):
    def __init__(self, in_channels=1, num_rrdb=23):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(num_rrdb)])
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feature = self.initial_conv(x)
        out = self.rrdb_blocks(initial_feature)
        out = self.final_conv(out)
        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# Sobel Loss
class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, sr, hr):
        # Apply Sobel filters
        sobel_x = self.sobel_x.to(sr.device)
        sobel_y = self.sobel_y.to(sr.device)

        # Compute gradients (edges) in x and y directions
        grad_sr_x = F.conv2d(sr, sobel_x, padding=1)
        grad_sr_y = F.conv2d(sr, sobel_y, padding=1)
        grad_hr_x = F.conv2d(hr, sobel_x, padding=1)
        grad_hr_y = F.conv2d(hr, sobel_y, padding=1)

        # Compute Sobel loss as the L1 distance between gradients
        grad_loss = F.l1_loss(grad_sr_x, grad_hr_x) + F.l1_loss(grad_sr_y, grad_hr_y)
        return grad_loss

# Loss functions
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, sr, hr):
        return F.mse_loss(sr, hr)

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg_model.features[:36]  # First 36 layers of VGG19
        self.vgg.eval()

    def forward(self, sr, hr):
        # Ensure both sr and hr have 3 channels for VGG (replicate grayscale)
        sr = sr.repeat(1, 3, 1, 1)  # Replicate grayscale to RGB
        hr = hr.repeat(1, 3, 1, 1)  # Replicate grayscale to RGB
        
        # Extract features using VGG
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        # Calculate MSE loss between feature maps
        return F.mse_loss(sr_features, hr_features)


import matplotlib.pyplot as plt

# Training function with loss tracking
def train(generator, discriminator, dataloader, num_epochs, optimizer_G, optimizer_D, criterion_content, criterion_perceptual, criterion_sobel, device):
    generator.to(device)
    discriminator.to(device)
    
    # Lists to store loss values for each epoch
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for i, (low_res_image, high_res_image) in enumerate(dataloader):
            low_res_image = low_res_image.to(device)
            high_res_image = high_res_image.to(device)

            # Generate output from the generator using the low-res image
            sr_image = generator(low_res_image)

            # Calculate content loss, perceptual loss, and sobel loss
            optimizer_G.zero_grad()
            content_loss = criterion_content(sr_image, high_res_image)  # Compare with high-res target
            perceptual_loss = criterion_perceptual(sr_image, high_res_image)
            sobel_loss = criterion_sobel(sr_image, high_res_image)
            g_loss = content_loss + perceptual_loss + sobel_loss
            g_loss.backward()
            optimizer_G.step()

            # Update discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(high_res_image)
            fake_output = discriminator(sr_image.detach())  # Detach generator output
            d_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output)) + \
                     F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
            d_loss.backward()
            optimizer_D.step()

            # Accumulate losses for the current epoch
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Step {i}, G Loss: {g_loss.item()}, D Loss: {d_loss.item()}")

        # Average loss for the epoch
        g_losses.append(epoch_g_loss / len(dataloader))
        d_losses.append(epoch_d_loss / len(dataloader))

        # Save model checkpoint after each epoch
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

    # Plot the graph of loss vs. epoch after training
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), g_losses, label="Generator Loss", color="b")
    plt.plot(range(num_epochs), d_losses, label="Discriminator Loss", color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main training script (remains the same)
if __name__ == "__main__":
    num_epochs = int(input("Enter number of epochs: "))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Use the modified dataset
    dataset = ImageDataset(low_res_dir="dataset/low_res", high_res_dir="dataset/high_res", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    vgg = models.vgg19(pretrained=True).to(device)
    criterion_content = ContentLoss()
    criterion_perceptual = PerceptualLoss(vgg)
    criterion_sobel = SobelLoss()

    train(generator, discriminator, dataloader, num_epochs, optimizer_G=optimizer_G, optimizer_D=optimizer_D,
          criterion_content=criterion_content, criterion_perceptual=criterion_perceptual, criterion_sobel=criterion_sobel, device=device)

